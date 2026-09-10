"""Tests for the documentation notebook path guard."""

import json

import pytest

from scripts.check_docs_notebook_paths import find_leaks


def _write_notebook(path, output: str, source: str = "print('example')") -> None:
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
                        "source": [source],
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
    assert leaks["posix.ipynb"] == ["/Users/example"]
    assert leaks["windows.ipynb"] == [
        r"C:\Users\example",
        r"C:\Users\example\project\.venv\warning.py:1",
    ]


def test_find_leaks_reports_identifying_paths_outside_outputs(tmp_path):
    """Catch user-home paths in source or metadata, not only saved outputs."""
    _write_notebook(
        tmp_path / "source.ipynb",
        "portable output",
        source="open('/home/example/private-data.csv')",
    )

    assert find_leaks(tmp_path) == {"source.ipynb": ["/home/example"]}


@pytest.mark.parametrize(
    "output",
    [
        "/home/runner/work/project/warning.py:1",
        "/private/var/folders/ab/session/T/notebook.py:1",
        "/tmp/build/project/warning.py:1",
        "/root/.cache/tool/state.json",
        "/workspace/project/warning.py:1",
        "/mnt/c/Users/example/project/warning.py:1",
        "file:///opt/project/build.log",
        r"D:\a\project\warning.py:1",
        r"\\server\share\project\warning.py:1",
    ],
)
def test_find_leaks_reports_common_local_output_roots(tmp_path, output):
    """Report common local, CI, temporary, Windows, and file-URI roots."""
    _write_notebook(tmp_path / "local.ipynb", output)

    assert find_leaks(tmp_path)["local.ipynb"]


def test_find_leaks_ignores_portable_paths_and_checkpoints(tmp_path):
    """Accept portable placeholders and ignore notebook checkpoints."""
    _write_notebook(
        tmp_path / "portable.ipynb",
        "<environment>/site-packages/warning.py:1 and /usr/local/lib/tool.so",
        source="write('/tmp/portable-example.txt')",
    )
    _write_notebook(
        tmp_path / ".ipynb_checkpoints" / "ignored.ipynb",
        "/home/example/project/.venv/warning.py:1",
    )

    assert find_leaks(tmp_path) == {}


def test_find_leaks_skips_binary_mime_payloads(tmp_path):
    """Do not mistake random base64 substrings in embedded figures for paths."""
    notebook = tmp_path / "figure.ipynb"
    content = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": 1,
                "metadata": {},
                "outputs": [
                    {
                        "data": {
                            "image/png": "random/tmp/base64-like-payload",
                            "text/plain": "<Figure size 640x480>",
                        },
                        "metadata": {},
                        "output_type": "display_data",
                    }
                ],
                "source": ["plot()"],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook.write_text(json.dumps(content))

    assert find_leaks(tmp_path) == {}

    content["cells"][0]["outputs"][0]["data"]["text/plain"] = (
        "/workspace/project/figure.png"
    )
    notebook.write_text(json.dumps(content))

    assert find_leaks(tmp_path) == {"figure.ipynb": ["/workspace/project"]}
