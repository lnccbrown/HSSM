"""Tests for the documentation notebook path guard."""

import json

from scripts.check_docs_notebook_paths import find_leaks


def _write_notebook(path, output: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cells": [
                    {
                        "cell_type": "code",
                        "execution_count": 1,
                        "metadata": {},
                        "outputs": [
                            {
                                "name": "stderr",
                                "output_type": "stream",
                                "text": output,
                            }
                        ],
                        "source": ["print('example')"],
                    }
                ],
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            }
        )
    )


def test_find_leaks_reports_posix_and_windows_user_paths(tmp_path):
    """Report user-specific paths on POSIX and Windows."""
    _write_notebook(
        tmp_path / "posix.ipynb", "/Users/example/project/.venv/warning.py:1"
    )
    _write_notebook(
        tmp_path / "windows.ipynb", r"C:\Users\example\project\.venv\warning.py:1"
    )

    leaks = find_leaks(tmp_path)

    assert set(leaks) == {"posix.ipynb", "windows.ipynb"}
    assert leaks["posix.ipynb"] == ["/Users/example/project/.venv/warning.py:1"]
    assert leaks["windows.ipynb"] == [
        r"C:\\Users\\example\\project\\.venv\\warning.py:1"
    ]


def test_find_leaks_ignores_portable_paths_and_checkpoints(tmp_path):
    """Accept portable placeholders and ignore notebook checkpoints."""
    _write_notebook(
        tmp_path / "portable.ipynb", "<environment>/site-packages/warning.py:1"
    )
    _write_notebook(
        tmp_path / ".ipynb_checkpoints" / "ignored.ipynb",
        "/home/example/project/.venv/warning.py:1",
    )

    assert find_leaks(tmp_path) == {}
