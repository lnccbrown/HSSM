# HSSM — Project Context for Claude

## What is HSSM?

HSSM (Hierarchical Sequential Sampling Models) is a Python package for Bayesian inference on sequential sampling models (DDM, LBA, etc.) using PyMC and bambi. It provides a high-level API for defining, fitting, and analyzing these models. This is the user-facing package in the HSSM ecosystem. For ecosystem-wide context, see the HSSMSpine repo.

## Project Structure

```
src/hssm/          # Main package source code
  hssm.py          # Core HSSM class — the main user-facing API
  config.py        # Model configuration
  modelconfig/     # 20+ model config files (DDM, LBA, RDM, etc.)
  likelihoods/     # Likelihood functions
  distribution_utils/
  param/           # Parameter handling
  plotting/        # Plotting utilities
  rl/              # Reinforcement learning SSM models
tests/             # pytest test suite
docs/              # MkDocs documentation source
  tutorials/       # Jupyter notebook tutorials (30+ notebooks)
  changelog.md     # Release changelog
  overrides/       # MkDocs theme overrides (banner, etc.)
.github/workflows/ # CI workflows
.claude/skills/    # Claude Code skills (e.g. prepare-release)
```

## Build & Tooling

- **Build system:** hatchling
- **Package manager:** uv (with `uv.lock`)
- **Python:** >=3.11, <3.14
- **Linting/formatting:** ruff (via pre-commit)
- **Type checking:** mypy
- **Pre-commit hooks:** end-of-file-fixer, trailing-whitespace, ruff, ruff-format, mypy

## Dependency Management

HSSM uses PEP 735 dependency groups alongside traditional optional-dependencies:

- **Core deps** (`[project.dependencies]`): pymc, bambi, numpyro, jax, arviz, etc.
- **Optional extras** (`[project.optional-dependencies]`): `cuda12`
- **Dependency groups** (`[dependency-groups]`):
  - `dev` — pytest, ruff, mypy, pre-commit, coverage
  - `notebook` — jupyterlab, nbconvert, graphviz, bayesflow (dev branch), keras
  - `docs` — mkdocs-material, mkdocs-jupyter, mkdocstrings-python

## Common Commands

```bash
# Install all dev dependencies
uv sync --group dev --group notebook --group docs

# Run tests
uv run pytest tests/

# Run slow tests (marked with @pytest.mark.slow)
uv run pytest tests/ --runslow

# Build docs
uv run --group notebook --group docs mkdocs build

# Serve docs locally
uv run --group notebook --group docs mkdocs serve

# Execute a single notebook (for verification)
uv run --group notebook jupyter nbconvert --ExecutePreprocessor.timeout=10000 --to notebook --execute docs/tutorials/<notebook>.ipynb

# Lint & format
uv run ruff check .
uv run ruff format .
```

## Key Patterns

### HSSM's `**kwargs` passthrough

`HSSM.sample()` passes `**kwargs` through to `bambi.Model.fit()`, which in turn passes them to PyMC's `pm.sample()`. So parameters like `cores`, `chains`, `nuts_sampler`, `target_accept`, etc. are valid even though they don't appear in HSSM's own signature. Similarly, the HSSM constructor passes `**kwargs` to `bambi.Model()`, so bambi parameters like `noncentered` are valid.

### ONNX likelihoods are single-trial + `jax.vmap`

Every ONNX graph consumed by HSSM must be exported with a concrete single-trial input shape (no `dynamic_axes`). Per-trial batching happens at the HSSM layer via `jax.vmap` over trials — see [`src/hssm/distribution_utils/onnx.py:115-138`](src/hssm/distribution_utils/onnx.py#L115-L138), where `logp(*inputs)` builds one flat per-trial vector and `make_vmap_func` lifts it.

Enforced at load time by `_check_single_trial_input_shape` in [`src/hssm/distribution_utils/onnx_utils/onnx2jax.py`](src/hssm/distribution_utils/onnx_utils/onnx2jax.py), which raises a `ValueError` on any symbolic input dim. The constraint exists because `jaxonnxruntime` traces against the construction-time dummy and bakes those shapes into the returned closure — calling that closure at a different batch size silently produces wrong outputs for graphs with batch-dependent intermediates (log-det accumulators, `Reshape` with `-1`).

LANfactory's exporters (`transform_sbi_to_onnx`, BayesFlow LRE export) already follow this convention. A new ONNX source must do the same: trace with a rank-1 dummy, no `dynamic_axes`.

### Notebook execution in CI

Two separate skip mechanisms for notebooks:
1. **mkdocs** `execute_ignore` in `mkdocs.yml` — skips execution during docs build
2. **CI** `SKIP_NOTEBOOKS` in `.github/workflows/check_notebooks.yml` — skips during notebook CI checks

### Versioning conventions

- `pyproject.toml` version: no prefix (e.g., `0.3.0`)
- Git tags: `v` prefix (e.g., `v0.3.0`)
- Changelog headings: no prefix (e.g., `### 0.3.0`)

## CI Workflows

| Workflow | Purpose |
|----------|---------|
| `run_tests.yml` | Fast test suite |
| `run_slow_tests.yml` | Slow tests (`@pytest.mark.slow`) |
| `linting_and_type_checking.yml` | ruff + mypy |
| `check_notebooks.yml` | Execute all non-skipped notebooks |
| `coverage.yml` | Code coverage |
| `build_docs.yml` | Build documentation |
| `build_and_publish.yml` | Release to PyPI (triggered on release publish) |
| `prepare-release.yml` | Release preparation automation |

## Known Issues / Notes

- **Multiprocessing in notebooks:** PyMC's NUTS sampler with `cores>1` can cause `EOFError` in notebook execution contexts. Fix by adding `cores=1, chains=1` to `sample()` calls.
- **bayesflow:** Currently installed from dev branch (`git+https://github.com/bayesflow-org/bayesflow@dev`) in the notebook group only, because PyPI release (2.0.8) doesn't yet include `RatioApproximator`. Not exposed as a user-facing optional extra until a stable release is available.
- **Notebook kernels:** Some notebooks may have hardcoded kernel specs (e.g., `hssm-dev`). These should use `python3` instead.
