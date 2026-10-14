"""Shared helpers for group-specific parameterization settings."""

from __future__ import annotations

from typing import TypeAlias

NoncenteredSetting: TypeAlias = bool | dict[str, bool] | None


def _resolve_noncentered(
    noncentered: NoncenteredSetting,
    component_name: str,
    prior_noncentered: bool | None = None,
) -> bool:
    """Compute the effective ``noncentered`` flag for one group term.

    This mirrors Bambi's resolution order in ``Model._set_priors``:

    1. A per-prior ``noncentered`` override takes precedence.
    2. Otherwise the model-level value is used, which may be a dictionary
       keyed by distributional component name.
    3. Missing dictionary keys fall back to Bambi's ``True`` default.

    Omission is captured by HSSM as ``True`` before this helper is called.
    An explicitly supplied ``None`` is passed through Bambi and is falsey in
    its backend, so it resolves to centered here.
    """
    if prior_noncentered is not None:
        return prior_noncentered
    if isinstance(noncentered, dict):
        return noncentered.get(component_name, True)
    if noncentered is None:
        return False
    return noncentered
