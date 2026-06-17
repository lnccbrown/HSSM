"""Layer 3: compose the emission (L2) and forward recursion (L1).

``make_hmm_logp_op`` closes over the panel shape, ``K``, the switching-param
identity, and the resolved emission distribution.  It returns a *model-builder
closure* that, given the regime parameter tensors plus ``log_P`` / ``log_pi0``,
adds the joint-marginal ``pm.Potential`` to the active ``pm.Model`` — identical
for the analytical and (Phase 3) LAN emission backends.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import pymc as pm
import pytensor.tensor as pt

from .emissions import per_regime_emission_logp
from .forward import forward_log_marginal

if TYPE_CHECKING:
    import numpy as np


def make_hmm_logp_op(
    dist_class,
    data_padded: np.ndarray,
    mask: np.ndarray,
    K: int,
    n_participants: int,
    n_trials: int,
    regime_params: set[str],
    pooling: str = "full",
    potential_name: str = "hmm_loglik",
) -> Callable[..., pt.TensorVariable]:
    """Build the forward-marginal model-builder closure.

    Parameters
    ----------
    dist_class
        The per-regime emission distribution class (from L2).
    data_padded
        ``(N, T, n_cols)`` padded panel data.
    mask
        ``(N, T)`` emission mask (1.0 real, 0.0 padded).
    K, n_participants, n_trials
        Panel/regime dimensions.
    regime_params
        Names of parameters carrying a regime axis (switching + fixed-per-regime).
    pooling
        ``"full"`` (global params) or ``"none"`` (per-participant params).
    potential_name
        Name of the ``pm.Potential`` added to the model.

    Returns
    -------
    Callable
        ``builder(param_values, log_P, log_pi0)`` which adds the
        ``pm.Potential`` to the active model and returns its tensor.
    """
    N, T = n_participants, n_trials
    n_cols = data_padded.shape[-1]
    data_flat_np = data_padded.reshape(N * T, n_cols).astype("float32")
    mask_np = mask.astype("float32")

    def builder(
        param_values: dict[str, pt.TensorVariable],
        log_P: pt.TensorVariable,
        log_pi0: pt.TensorVariable,
    ) -> pt.TensorVariable:
        data_flat = pt.as_tensor_variable(data_flat_np, name="hmm_data")

        regime_param_dicts: list[dict[str, pt.TensorVariable]] = []
        for k in range(K):
            params_k: dict[str, pt.TensorVariable] = {}
            for name, val in param_values.items():
                has_regime = name in regime_params
                if pooling == "full":
                    # switching/fixed-per-regime: (K,) -> scalar val[k];
                    # shared: scalar broadcast.
                    params_k[name] = val[k] if has_regime else val
                else:  # "none": per-participant params
                    # switching: (N, K) -> column (N,); shared: (N,).
                    col = val[:, k] if has_regime else val
                    # Expand each participant's value across its T trials to
                    # align with the (participant-major, trial-minor) data rows.
                    params_k[name] = pt.repeat(col, T)
            regime_param_dicts.append(params_k)

        emission_flat = per_regime_emission_logp(
            dist_class, data_flat, regime_param_dicts
        )  # (M, K)
        log_emission = emission_flat.reshape((N, T, K))

        mask_t = pt.as_tensor_variable(mask_np, name="hmm_mask")
        marginal = forward_log_marginal(log_emission, log_P, log_pi0, mask_t)
        return pm.Potential(potential_name, marginal)

    return builder
