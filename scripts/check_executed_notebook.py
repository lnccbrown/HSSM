#!/usr/bin/env python3
"""Fail when a retained executed notebook contains error outputs."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any


class ExecutedNotebookError(ValueError):
    """Raised when a retained artifact is not a readable notebook."""


def execution_errors(notebook: pathlib.Path) -> list[str]:
    """Return ordered summaries for all code-cell error outputs."""
    try:
        content: Any = json.loads(notebook.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ExecutedNotebookError(f"cannot read {notebook}: {error}") from error
    if not isinstance(content, dict):
        raise ExecutedNotebookError(f"{notebook} notebook root is not a JSON object")

    cells = content.get("cells")
    if not isinstance(cells, list):
        raise ExecutedNotebookError(f"{notebook} has no notebook cell list")

    errors: list[str] = []
    for index, cell in enumerate(cells, start=1):
        if not isinstance(cell, dict) or cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        if not isinstance(outputs, list):
            raise ExecutedNotebookError(
                f"{notebook} code cell {index} has an invalid output list"
            )
        for output in outputs:
            if not isinstance(output, dict) or output.get("output_type") != "error":
                continue
            error_name = output.get("ename", "Error")
            error_value = output.get("evalue", "")
            summary = f"cell {index}: {error_name}"
            if error_value:
                summary += f": {error_value}"
            errors.append(summary)
    return errors


def main(argv: list[str] | None = None) -> int:
    """Report retained execution errors for one notebook."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notebook", type=pathlib.Path)
    args = parser.parse_args(argv)

    try:
        errors = execution_errors(args.notebook)
    except ExecutedNotebookError as error:
        print(error, file=sys.stderr)
        return 2

    if errors:
        for error in errors:
            print(f"EXECUTION ERROR: {error}")
        return 1

    print(f"ok: no execution errors in {args.notebook}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
