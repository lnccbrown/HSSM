"""Layer 2: per-regime SSM emission log-density.

L2 resolves a per-regime emission callable.  In v1-Phase-2 the analytical
likelihood path is implemented; the LAN (``approx_differentiable``) path is a
Phase-3 addition that resolves a different callable here while composing into
the *same* forward recursion (design doc §5.2, decision 10.1.6).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pymc as pm
import pytensor.tensor as pt

from ...distribution_utils import make_distribution_for_supported_model

if TYPE_CHECKING:
    from ..._types import LoglikKind, SupportedModels


def resolve_emission_dist(
    model: str,
    loglik_kind: "LoglikKind",
    backend: str | None,
    is_choice_only: bool = False,
):
    """Resolve the per-regime SSM distribution class for the emission.

    Returns a ``pm.Distribution`` subclass whose ``.dist(**params)`` accepts the
    SSM parameters and whose ``pm.logp(dist, data)`` evaluates the per-trial
    log-density (the same object the hand-written tutorial uses).
    """
    resolved_backend = backend if backend is not None else "pytensor"
    return make_distribution_for_supported_model(
        model=cast("SupportedModels", model),
        loglik_kind=loglik_kind,
        backend=resolved_backend,  # type: ignore[arg-type]
        is_choice_only=is_choice_only,
    )


def per_regime_emission_logp(
    dist_class,
    data_flat: pt.TensorVariable,
    regime_param_dicts: list[dict[str, pt.TensorVariable]],
) -> pt.TensorVariable:
    """Evaluate the SSM logp once per regime and stack.

    Parameters
    ----------
    dist_class
        The emission distribution class from :func:`resolve_emission_dist`.
    data_flat
        ``(M, n_cols)`` flattened panel data (``M = N * T``).
    regime_param_dicts
        Length-K list; entry ``k`` maps every SSM parameter name to that
        regime's scalar value.

    Returns
    -------
    pt.TensorVariable
        ``(M, K)`` per-trial, per-regime emission log-density.
    """
    components = [
        pm.logp(dist_class.dist(**params), data_flat) for params in regime_param_dicts
    ]
    return pt.stack(components, axis=1)
