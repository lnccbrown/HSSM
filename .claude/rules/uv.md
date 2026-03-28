---
description: Enforce uv as the package manager for all Python operations
globs:
  - "**/*.py"
  - "**/pyproject.toml"
---

- Always use `uv run` to execute commands — never bare `python`, `pytest`, `ruff`, or other tools.
- Never use `pip install` — use `uv sync` (with `--group` flags) to manage dependencies.
- The `uv.lock` file is the source of truth for resolved dependency versions.
- When adding dependencies, add them to `pyproject.toml` and run `uv sync`.
- HSSM uses PEP 735 dependency groups: `--group dev`, `--group notebook`, `--group docs`.
