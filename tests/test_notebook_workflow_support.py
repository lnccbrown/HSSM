"""Tests for safe targeted notebook execution support."""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile
import unittest

from scripts.check_executed_notebook import (
    ExecutedNotebookError,
    execution_errors,
)
from scripts.validate_notebook_target import (
    NotebookTargetError,
    validate_notebook_target,
)


def _write_notebook(path: pathlib.Path, outputs: list[dict] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "metadata": {},
                        "outputs": outputs or [],
                        "source": ["print('ok')"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )


class NotebookTargetTests(unittest.TestCase):
    """Validate the target boundary before a workflow executes code."""

    def setUp(self) -> None:
        """Create a temporary repository with one tracked notebook."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self.temp_dir.name)
        subprocess.run(
            ["git", "init", "--quiet", str(self.root)],
            check=True,
            capture_output=True,
            text=True,
        )
        self.good = self.root / "docs" / "tutorials" / "good.ipynb"
        self.skipped = self.root / "docs" / "tutorials" / "skipped.ipynb"
        _write_notebook(self.good)
        _write_notebook(self.skipped)
        skip_list = self.root / ".github" / "notebook-skip-list.txt"
        skip_list.parent.mkdir(parents=True)
        skip_list.write_text("docs/tutorials/skipped.ipynb\n")
        subprocess.run(
            [
                "git",
                "-C",
                str(self.root),
                "add",
                "docs/tutorials/good.ipynb",
                "docs/tutorials/skipped.ipynb",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    def tearDown(self) -> None:
        """Remove the temporary repository."""
        self.temp_dir.cleanup()

    def test_accepts_one_tracked_non_archive_notebook(self) -> None:
        """Accept a canonical tracked tutorial path."""
        target = validate_notebook_target(
            "docs/tutorials/good.ipynb", repo_root=self.root
        )

        self.assertEqual(target, self.good.resolve())

    def test_rejects_an_untracked_notebook(self) -> None:
        """Do not execute files that are absent from the git index."""
        _write_notebook(self.root / "docs" / "tutorials" / "untracked.ipynb")

        with self.assertRaisesRegex(NotebookTargetError, "not tracked"):
            validate_notebook_target(
                "docs/tutorials/untracked.ipynb", repo_root=self.root
            )

    def test_rejects_a_missing_notebook(self) -> None:
        """Reject a tracked path when its working-tree file is absent."""
        self.good.unlink()

        with self.assertRaisesRegex(NotebookTargetError, "does not exist"):
            validate_notebook_target("docs/tutorials/good.ipynb", repo_root=self.root)

    def test_rejects_a_notebook_disabled_by_ci_policy(self) -> None:
        """Apply the shared full-suite skip policy to targeted runs."""
        with self.assertRaisesRegex(NotebookTargetError, "CI skip policy"):
            validate_notebook_target(
                "docs/tutorials/skipped.ipynb", repo_root=self.root
            )

    def test_rejects_a_symlink_that_escapes_docs(self) -> None:
        """Resolve symlinks before enforcing the documentation root."""
        outside = self.root / "outside.ipynb"
        _write_notebook(outside)
        link = self.root / "docs" / "tutorials" / "escape.ipynb"
        link.symlink_to(outside)
        subprocess.run(
            ["git", "-C", str(self.root), "add", "docs/tutorials/escape.ipynb"],
            check=True,
            capture_output=True,
            text=True,
        )

        with self.assertRaisesRegex(NotebookTargetError, "escapes docs"):
            validate_notebook_target("docs/tutorials/escape.ipynb", repo_root=self.root)

    def test_rejects_unsafe_target_shapes(self) -> None:
        """Reject traversal, archives, controls, and non-notebook targets."""
        invalid_targets = {
            "": "empty",
            "/docs/tutorials/good.ipynb": "absolute",
            "docs/../outside.ipynb": "traversal",
            "docs//tutorials/good.ipynb": "canonical",
            "docs/tutorials\\good.ipynb": "POSIX",
            "docs/tutorials/good.ipynb\n": "control",
            "docs/archive/snapshot.ipynb": "archived",
            "docs/.ipynb_checkpoints/good.ipynb": "checkpoints",
            "tests/good.ipynb": "below docs",
            "docs/tutorials/good.py": "ipynb",
        }

        for target, message in invalid_targets.items():
            with self.subTest(target=target):
                with self.assertRaisesRegex(NotebookTargetError, message):
                    validate_notebook_target(target, repo_root=self.root)


class ExecutedNotebookTests(unittest.TestCase):
    """Retain failed outputs while still making the hosted job fail."""

    def test_accepts_a_notebook_without_error_outputs(self) -> None:
        """Treat ordinary stream output as a successful execution."""
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook = pathlib.Path(temp_dir) / "executed.ipynb"
            _write_notebook(
                notebook,
                [{"name": "stdout", "output_type": "stream", "text": ["ok\n"]}],
            )

            self.assertEqual(execution_errors(notebook), [])

    def test_reports_every_error_output_in_cell_order(self) -> None:
        """Summarize every saved error without discarding the artifact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook = pathlib.Path(temp_dir) / "failed.ipynb"
            _write_notebook(
                notebook,
                [
                    {
                        "ename": "ValueError",
                        "evalue": "bad value",
                        "output_type": "error",
                        "traceback": ["traceback"],
                    },
                    {
                        "ename": "RuntimeError",
                        "evalue": "backend failed",
                        "output_type": "error",
                        "traceback": ["traceback"],
                    },
                ],
            )

            self.assertEqual(
                execution_errors(notebook),
                [
                    "cell 1: ValueError: bad value",
                    "cell 1: RuntimeError: backend failed",
                ],
            )

    def test_rejects_malformed_notebook_json(self) -> None:
        """Report malformed retained output as an infrastructure failure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook = pathlib.Path(temp_dir) / "broken.ipynb"
            notebook.write_text("not json")

            with self.assertRaisesRegex(ExecutedNotebookError, "cannot read"):
                execution_errors(notebook)

    def test_rejects_a_non_object_notebook_root(self) -> None:
        """Return a controlled error for valid JSON with the wrong root type."""
        with tempfile.TemporaryDirectory() as temp_dir:
            notebook = pathlib.Path(temp_dir) / "list.ipynb"
            notebook.write_text("[]")

            with self.assertRaisesRegex(ExecutedNotebookError, "not a JSON object"):
                execution_errors(notebook)


if __name__ == "__main__":
    unittest.main()
