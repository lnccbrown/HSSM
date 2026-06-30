"""Likelihood builder for the aDDM.

Mirrors :mod:`hssm.rl.likelihoods.builder` (``make_rl_logp_func`` /
``make_rl_logp_op``): it maps HSSM's column-ordering contract onto the vendored
batched JAX kernel and wraps the result as a differentiable PyTensor ``Op`` for
NUTS, reusing the shared JAX->PyTensor machinery (no new Op infrastructure).

Column-ordering contract
------------------------
At model-build time the distribution calls ``loglik(data, *dist_params,
*extra_fields)`` where ``dist_params`` are ``list_params`` in order and
``extra_fields`` follow ``model_config.extra_fields`` order. For aDDM:

- ``data[:, 0]`` = rt, ``data[:, 1]`` = response
- ``list_params``  = ``[eta, kappa, a, b, x0]``
- ``extra_fields`` = ``[r1, r2, flag, sacc_array, d, sigma]``

so ``logp`` receives ``(data, eta, kappa, a, b, x0, r1, r2, flag, sacc_array, d,
sigma)`` and reorders them into the kernel's positional slots (note ``sigma`` is
the last extra field but the 5th kernel argument).

Unlike RLSSM, aDDM has no ``ssm_logp_func.computed`` panel machinery and no
``n_participants``/``n_trials`` reshaping — each trial's likelihood depends only
on that trial's own covariates.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

from hssm.distribution_utils.func_utils import make_vjp_func
from hssm.distribution_utils.jax import make_jax_logp_ops

from ..attention_process import resolve_attention_process, standard_alternating
from .jax import compute_addm_loglikelihoods, compute_addm_loglikelihoods_from_mu

if TYPE_CHECKING:
    from pytensor.graph import Op


def make_addm_logp_func(
    attention_process: str | Callable = "standard_alternating",
) -> Callable:
    """Build the per-trial aDDM log-likelihood closure.

    Parameters
    ----------
    attention_process
        Registry name or callable producing the per-trial drift array. The
        default ``"standard_alternating"`` takes the fast path that lets the
        kernel build the drift array internally; any other process is invoked to
        produce ``mu`` which is then fed to ``compute_addm_loglikelihoods_from_mu``.

    Returns
    -------
    Callable
        ``logp(data, eta, kappa, a, b, x0, r1, r2, flag, sacc_array, d, sigma)``
        returning per-trial log-likelihoods of shape ``(n_trials,)``.
    """
    process = resolve_attention_process(attention_process)
    use_default = process is standard_alternating

    def logp(data, eta, kappa, a, b, x0, r1, r2, flag, sacc_array, d, sigma):
        rt = data[:, 0]
        response = data[:, 1]
        # sigma is the (fixed) diffusion-noise constant. The kernel vmaps over
        # trials but treats sigma as a scalar, so a per-trial sigma column would
        # collide with the quadrature grid. Reduce to a scalar (validated constant
        # upstream); a scalar/0-d sigma passes through unchanged.
        sigma = jnp.asarray(sigma).reshape(-1)[0]
        if use_default:
            # Kernel builds the drift array internally via the same
            # _build_addm_mu_array_data the default attention process delegates to.
            return compute_addm_loglikelihoods(
                rt, response, eta, kappa, sigma, a, b, x0,
                r1, r2, flag, sacc_array, d,
            )
        mu = process(eta, kappa, r1, r2, flag, d, sacc_array.shape[1])
        return compute_addm_loglikelihoods_from_mu(
            rt, response, mu, sacc_array, d, sigma, a, b, x0,
        )

    return logp


def make_addm_logp_op(
    attention_process: str | Callable,
    list_params: list[str],
    extra_fields: list[str],
) -> "Op":
    """Wrap the aDDM JAX log-likelihood as a differentiable PyTensor ``Op``.

    Uses the same construction as ``make_rl_logp_op``: build the JAX ``logp``,
    derive its VJP, and wrap both in a PyTensor ``Op`` whose gradient is wired to
    the VJP. ``n_params = len(list_params)`` ensures gradients are computed only
    for the sampled parameters; gradients w.r.t. the extra fields are left
    undefined.

    Parameters
    ----------
    attention_process
        Registry name or callable (see :func:`make_addm_logp_func`).
    list_params
        Sampled parameter names, in order (e.g. ``["eta", "kappa", "a", "b", "x0"]``).
    extra_fields
        Per-trial covariate names, in order. Used only for its length / contract;
        the values are supplied at call time by HSSM's extra-fields machinery.

    Returns
    -------
    Op
        A PyTensor ``Op`` usable with ``pytensor.grad`` (NUTS).
    """
    logp = make_addm_logp_func(attention_process)
    n_params = len(list_params or [])
    vjp_logp = make_vjp_func(logp, params_only=False, n_params=n_params)

    return make_jax_logp_ops(
        logp=jax.jit(logp),
        logp_vjp=jax.jit(vjp_logp),
        logp_nojit=logp,
        n_params=n_params,
    )
