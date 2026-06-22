# Vendored from efficient-fpt @ d97a451; do not edit in place — re-vendor instead.
"""JAX-specific ADDM drift-rate array construction helpers.

Mirrors the NumPy ``addm_helpers.py`` but uses ``jax.numpy`` for
GPU acceleration and automatic differentiation compatibility.
"""

import jax.numpy as jnp
from jax import vmap

from .utils import get_jax_dtype


def _build_addm_mu_array(eta, kappa, r1, r2, flag, d, max_d, dtype):
    """Return the padded single-trial ADDM drift array from user-facing covariates."""
    mu1 = kappa * (r1 - eta * r2)
    mu2 = kappa * (eta * r1 - r2)
    mu_first = jnp.where(flag == 0, mu1, mu2)
    mu_second = jnp.where(flag == 0, mu2, mu1)
    stage_idx = jnp.arange(max_d, dtype=jnp.int32)
    mu_array = jnp.where(stage_idx % 2 == 0, mu_first, mu_second)
    return jnp.where(stage_idx < d, mu_array, jnp.zeros((), dtype=dtype))


def _build_addm_mu_array_data(eta, kappa, r1_data, r2_data, flag_data, d_data, max_d):
    """Build the padded addm drift array from user-facing batch covariates."""
    dtype = jnp.result_type(r1_data, r2_data, get_jax_dtype())
    trial_mu = vmap(
        lambda r1, r2, flag, d: _build_addm_mu_array(
            eta, kappa, r1, r2, flag, d, max_d, dtype,
        )
    )
    return trial_mu(r1_data, r2_data, flag_data, d_data)
