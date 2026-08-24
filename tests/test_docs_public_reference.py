"""Keep reader-facing model and API reference aligned with public code."""

from __future__ import annotations

import re
from pathlib import Path

from hssm.config import Config
from hssm.modelconfig import get_default_model_config, list_models

ROOT = Path(__file__).parent.parent
MODEL_REFERENCE = ROOT / "docs/reference/models-and-likelihoods.md"


def _code_values(cell: str) -> tuple[str, ...]:
    """Return the ordered code spans in one Markdown table cell."""
    return tuple(re.findall(r"`([^`]+)`", cell))


def _model_rows() -> tuple[
    tuple[str, tuple[str, ...], str, tuple[str, ...], tuple[str, ...]], ...
]:
    rows: list[tuple[str, tuple[str, ...], str, tuple[str, ...], tuple[str, ...]]] = []
    for line in MODEL_REFERENCE.read_text().splitlines():
        cells = tuple(cell.strip() for cell in line.strip().strip("|").split("|"))
        if len(cells) != 5 or not cells[0].startswith("`"):
            continue
        model_values = _code_values(cells[0])
        default_values = _code_values(cells[2])
        assert len(model_values) == len(default_values) == 1
        rows.append(
            (
                model_values[0],
                _code_values(cells[1]),
                default_values[0],
                _code_values(cells[3]),
                _code_values(cells[4]),
            )
        )
    return tuple(rows)


def test_builtin_model_matrix_matches_defaults() -> None:
    """Require the documented built-in matrix to match current configurations."""
    rows = _model_rows()
    models = tuple(list_models())
    assert tuple(row[0] for row in rows) == models
    assert len({row[0] for row in rows}) == len(rows)

    for model, kinds, default, params, choices in rows:
        config = get_default_model_config(model)
        runtime_default = Config.from_defaults(model, None)
        assert runtime_default is not None
        assert (kinds, default, params, choices) == (
            tuple(config["likelihoods"]),
            runtime_default.loglik_kind,
            tuple(config["list_params"]),
            tuple(str(choice) for choice in config["choices"]),
        )


def test_previously_missing_top_level_apis_are_documented() -> None:
    """Keep specialized top-level public exports in the API reference."""
    expectations = {
        "docs/api/addm.md": ("hssm.aDDM", "hssm.aDDMConfig"),
        "docs/api/model_registry.md": ("hssm.list_models", "hssm.register_model"),
    }
    for relative_path, names in expectations.items():
        contents = (ROOT / relative_path).read_text()
        assert all(name in contents for name in names)


def test_onnx_reference_states_the_enforced_contract() -> None:
    """Keep the central ONNX invariants visible on the canonical page."""
    contract = " ".join(
        (ROOT / "docs/how_to/custom_onnx_likelihoods.md").read_text().split()
    )
    for required in (
        "every input dimension must be a concrete integer",
        "Symbolic dimensions and `dynamic_axes` are forbidden",
        "HSSM batches the per-trial function itself with `jax.vmap`",
        "model parameters in `list_params` order",
    ):
        assert required in contract
