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
- ``list_params``  = ``[eta, kappa, a, b, x0, t]``
- ``extra_fields`` = ``[r1, r2, flag, sacc_array, d, sigma]``

so ``logp`` receives ``(data, eta, kappa, a, b, x0, t, r1, r2, flag, sacc_array,
d, sigma)`` and reorders them into the kernel's positional slots (note ``sigma``
is the last extra field but the 5th kernel argument, and ``t`` is a sampled
non-decision time applied as a decision-time shift, not a kernel argument).

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
from .jax import (
    compute_addm_logfptd,
    compute_addm_loglikelihoods,
    compute_addm_loglikelihoods_from_mu,
)

if TYPE_CHECKING:
    from pytensor.graph import Op

# Core params the kernel can take per-trial (regression / hierarchical), in the
# kernel's positional order. ``sigma`` is intentionally excluded — it stays a
# scalar diffusion constant (per-trial sigma would collide with the quadrature
# grid; the validator enforces this upstream).
_CORE_PARAMS = ("eta", "kappa", "a", "b", "x0")


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
        ``logp(data, eta, kappa, a, b, x0, t, r1, r2, flag, sacc_array, d, sigma)``
        returning per-trial log-likelihoods of shape ``(n_trials,)``.

    Notes
    -----
    Core params (``eta, kappa, a, b, x0``) may be either scalar (a shared prior)
    or per-trial (a regression / hierarchical prior). A regressed param reaches
    this closure as an ``(n_obs,)`` array — bambi's per-trial linear predictor,
    with dist.py's trial-wise broadcast guaranteeing the shape. When *all* core
    params are scalar we take the optimized batched kernel; when *any* is
    per-trial we vmap the single-trial kernel with ``in_axes=0`` on exactly the
    per-trial ones (the same trial-wise pattern as the ONNX likelihood path),
    which keeps the vendored kernel untouched.

    Non-decision time ``t`` (scalar or per-trial) is applied in this builder — not
    the vendored kernel — by shifting into *decision time* before any kernel call:
    ``rt_eff = rt - t`` and every fixation onset slides back by ``t`` with the
    first anchored at 0, so the only live effect is to shorten the first stage by
    ``t`` (``t=0`` is a bit-for-bit identity). A per-trial constraint (``t`` must
    fall inside the first fixation and leave ``rt - t > 0``) rejects offending
    trials with ``-inf``; the kernel is evaluated on a clamped, in-support ``t``
    so gradients stay finite even for rejected trials.
    """
    process = resolve_attention_process(attention_process)
    use_default = process is standard_alternating

    def logp(data, eta, kappa, a, b, x0, t, r1, r2, flag, sacc_array, d, sigma):
        rt = data[:, 0]
        response = data[:, 1]
        n_obs = rt.shape[0]
        # sigma is the (fixed) diffusion-noise constant — always reduced to a
        # scalar (a per-trial sigma would collide with the quadrature grid).
        sigma = jnp.asarray(sigma).reshape(-1)[0]

        # --- non-decision time t: shift into decision time (builder-side) ------
        # t may be scalar (shared) or (n_obs,) (regressed). Reject is computed
        # from the ORIGINAL t; the kernel is fed a clamped, in-support t_safe so
        # rejected trials never produce NaNs whose gradient could leak through the
        # jnp.where mask. First fixation ends at the 2nd onset (d>=2); a d==1 trial
        # has no interior fixation, so only the rt>t clause gates it.
        t = jnp.asarray(t)
        max_d = sacc_array.shape[1]
        first_fix_end = jnp.where(d >= 2, sacc_array[:, 1], jnp.inf)
        reject = (t < 0.0) | (t >= first_fix_end) | (rt - t <= 0.0)
        reject = jnp.broadcast_to(reject, (n_obs,))
        eps = 1e-6
        t_safe = jnp.clip(t, 0.0, first_fix_end - eps)  # -> (n_obs,) after broadcast
        rt_eff = rt - t_safe
        col = jnp.arange(max_d)
        valid_col = (col[None, :] >= 1) & (col[None, :] < d[:, None])
        sacc_eff = jnp.where(valid_col, sacc_array - t_safe[:, None], sacc_array)

        core = dict(eta=eta, kappa=kappa, a=a, b=b, x0=x0)

        def _is_trialwise(p):
            p = jnp.asarray(p)
            # ponytail: shape[0]==n_obs is the trial-wise signal; ambiguous only
            # for a 1-trial dataset (n_obs==1), which is not a real inference case.
            return p.ndim >= 1 and p.shape[0] == n_obs

        trialwise = {name: _is_trialwise(p) for name, p in core.items()}

        if not any(trialwise.values()):
            # Fast path: all core params scalar -> optimized batched kernel.
            eta, kappa, a, b, x0 = (
                jnp.asarray(core[name]).reshape(-1)[0] for name in _CORE_PARAMS
            )
            if use_default:
                # Kernel builds the drift array internally via the same
                # _build_addm_mu_array_data the default process delegates to.
                ll = compute_addm_loglikelihoods(
                    rt_eff,
                    response,
                    eta,
                    kappa,
                    sigma,
                    a,
                    b,
                    x0,
                    r1,
                    r2,
                    flag,
                    sacc_eff,
                    d,
                )
                return jnp.where(reject, -jnp.inf, ll)
            # Drift is built from the ORIGINAL covariates (unaffected by the
            # decision-time shift); only rt/sacc are shifted for the FPT kernel.
            mu = process(eta, kappa, r1, r2, flag, d, max_d)
            ll = compute_addm_loglikelihoods_from_mu(
                rt_eff,
                response,
                mu,
                sacc_eff,
                d,
                sigma,
                a,
                b,
                x0,
            )
            return jnp.where(reject, -jnp.inf, ll)

        if not use_default:
            raise NotImplementedError(
                "Trial-wise / regression core params are supported only with the "
                "default 'standard_alternating' attention process; a custom "
                "attention process builds the drift batch-wise from scalar "
                "eta/kappa. Regress under the default process, or extend the "
                "custom process to accept per-trial eta/kappa."
            )

        # Per-trial path: vmap the single-trial kernel, mapping only the per-trial
        # core params (in_axes=0) and passing the scalar ones unmapped (None).
        mapped = {
            name: (core[name] if tw else jnp.asarray(core[name]).reshape(-1)[0])
            for name, tw in trialwise.items()
        }
        in_axes = (
            0,
            0,  # rt, response
            *(0 if trialwise[name] else None for name in ("eta", "kappa")),
            None,  # sigma
            *(0 if trialwise[name] else None for name in ("a", "b", "x0")),
            0,
            0,
            0,
            0,
            0,  # r1, r2, flag, sacc_array, d
        )
        out = jax.vmap(compute_addm_logfptd, in_axes=in_axes)(
            rt_eff,
            response,
            mapped["eta"],
            mapped["kappa"],
            sigma,
            mapped["a"],
            mapped["b"],
            mapped["x0"],
            r1,
            r2,
            flag,
            sacc_eff,
            d,
        )
        return jnp.where(reject, -jnp.inf, out)

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
