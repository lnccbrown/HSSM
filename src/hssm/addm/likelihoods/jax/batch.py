# Vendored from efficient-fpt @ d97a451; do not edit in place — re-vendor instead.
# HSSM-authored additions to the vendored copy: above-package imports retargeted
# to siblings (``._defaults``, ``.utils``), and a public
# ``compute_addm_loglikelihoods_from_mu`` wrapper exposed near the bottom of this
# file (a stable entry over the private batchscan core for custom attention
# processes). Preserve both on re-vendor.
"""Batched aDDM log-likelihood and NLL computation using JAX kernels.

This module contains the JAX batch layer for aDDM log-likelihoods and NLLs. It
exposes two explicit batch methods:

- a legacy baseline that vmaps the production single-trial kernel in
  ``jax.multi_stage`` over trials
- a dedicated batched stage-scan kernel that carries the whole batch through
  one multistage recurrence

These batch variants now sit alongside the single-trial variants in
``jax.multi_stage``:

- ``jax.multi_stage.compute_*_precomputed``: public single-trial production
  kernels
- ``jax.multi_stage.compute_*_stagescan``: public single-trial stage-scan
  kernels
- ``jax.batch.compute_addm_loglikelihoods_batchvmap``: legacy batch baseline
- ``jax.batch.compute_addm_loglikelihoods_batchscan``: production batch kernel
- ``jax.batch.compute_addm_loglikelihoods``: alias to
  ``compute_addm_loglikelihoods_batchscan``
- ``jax.batch.make_addm_nll_function_batchvmap``: legacy optimizer closure
- ``jax.batch.make_addm_nll_function_batchscan``: production optimizer closure
- ``jax.batch.make_addm_nll_function``: alias to
  ``make_addm_nll_function_batchscan``

There is currently no batched kernel that precomputes *all* stage transition
matrices for the general current batch API up front. That design is possible
in principle, but because stage drifts and stage schedules vary by trial, the
fully general mid-stage transition tensor would scale roughly like
``(batch_size, max_d - 3, order_mid, order_mid)``, plus a separate
rectangular final transition into the ``order_last`` grid.

JAX uses a fixed-length series controlled only by ``trunc_num``. There is no
``adaptive_stopping`` or ``threshold`` option in this backend.
"""

import warnings

import jax.numpy as jnp
import numpy as np
from jax import jit, lax, remat, vmap
from jax.scipy.special import logsumexp

from ._defaults import (
    DEFAULT_LAST_QUAD_ORDER,
    DEFAULT_MID_QUAD_ORDER,
    DEFAULT_TRUNC_NUM,
)
from .addm_helpers import _build_addm_mu_array_data
from .multi_stage import (
    _reduce_final_stage_fptds,
    _symmetric_stage_grid,
    compute_addm_logfptd,
)
from .single_stage import fptd_single, log_fptd_single, q_single
from .utils import (
    _DUMMY_STAGE_DURATION,
    get_gauss_legendre_ref,
    positive_log,
    resolve_quadrature_orders,
)

# ---------------------------------------------------------------------------
# Warning and NLL reduction helpers
# ---------------------------------------------------------------------------


def _warn_invalid_loglikelihoods(loglikelihoods):
    """Emit deterministic warnings for bad trial log-likelihoods.

    Parameters
    ----------
    loglikelihoods : array-like, shape (n_trials,)
        Per-trial log-likelihood values. ``-inf`` and invalid values trigger
        deterministic warnings shared with the other backends.

    Notes
    -----
    This helper intentionally avoids ``np.asarray(loglikelihoods)`` on the
    full vector, which would force an unnecessary host transfer for large
    device-resident batches. Instead, it computes the bad-trial indices on the
    device and transfers only those compact index arrays to Python.
    """
    neginf_mask = jnp.isneginf(loglikelihoods)
    invalid_mask = ~(jnp.isfinite(loglikelihoods) | neginf_mask)

    neginf_idx = np.asarray(jnp.nonzero(neginf_mask)[0]).ravel()
    invalid_idx = np.asarray(jnp.nonzero(invalid_mask)[0]).ravel()

    for idx in neginf_idx:
        warnings.warn(
            f"trial {int(idx)} outputs -inf log-likelihood",
            RuntimeWarning,
            stacklevel=2,
        )
    for idx in invalid_idx:
        warnings.warn(
            f"trial {int(idx)} outputs invalid log-likelihood",
            RuntimeWarning,
            stacklevel=2,
        )


def _reduce_addm_loglikelihoods_to_nll(
    loglikelihoods, reduce="mean", invalid_policy="inf", warn=True
):
    """Reduce a log-likelihood vector to a mean or summed negative log-likelihood.

    Parameters
    ----------
    loglikelihoods : jax.Array, shape (n_trials,)
        Per-trial log-likelihood values.
    reduce : {"mean", "sum"}, optional
        Reduction to apply after filtering invalid trials.
    invalid_policy : {"inf", "warn"}, optional
        ``"inf"`` propagates ``-inf`` trials to ``+inf`` NLL and invalid
        trials to ``NaN``. ``"warn"`` emits warnings and skips all non-finite
        trials.
    warn : bool, optional
        Whether to emit deterministic Python warnings for bad trials.
    """
    finite = jnp.isfinite(loglikelihoods)
    neginf = jnp.isneginf(loglikelihoods)
    invalid = ~(finite | neginf)
    if warn:
        _warn_invalid_loglikelihoods(loglikelihoods)

    losses = jnp.where(finite, -loglikelihoods, 0.0)
    total_loss = jnp.sum(losses)
    num_valid = jnp.sum(finite.astype(jnp.int32))

    if invalid_policy == "inf":
        reduced = total_loss if reduce == "sum" else total_loss / num_valid
        reduced = jnp.where(num_valid > 0, reduced, jnp.nan)
        reduced = jnp.where(jnp.any(neginf), jnp.inf, reduced)
        return jnp.where(jnp.any(invalid), jnp.nan, reduced)

    if invalid_policy != "warn":
        raise ValueError(
            f"invalid_policy must be 'inf' or 'warn', got {invalid_policy!r}"
        )

    if reduce == "sum":
        return jnp.where(num_valid > 0, total_loss, jnp.nan)
    return jnp.where(num_valid > 0, total_loss / num_valid, jnp.nan)


# ---------------------------------------------------------------------------
# Batched schedule builders
# ---------------------------------------------------------------------------


def _batch_addm_first_stage_to_grid(
    xs_target,
    ws_target,
    *,
    mu0_data,
    sigma,
    a,
    upper_slope0_data,
    lower_slope0_data,
    stage_duration0_data,
    x0,
    trunc_num,
    log_space,
):
    """Initialize batched first-stage weighted alive-state mass on a grid."""
    pv_init = vmap(
        lambda xs, mu, upper_slope, lower_slope, dt: q_single(
            xs,
            mu,
            sigma,
            a,
            upper_slope,
            -a,
            lower_slope,
            dt,
            x0,
            trunc_num=trunc_num,
        )
    )(
        xs_target,
        mu0_data,
        upper_slope0_data,
        lower_slope0_data,
        stage_duration0_data,
    )
    ws_pv = ws_target * pv_init
    return positive_log(ws_pv) if log_space else ws_pv


def _safe_stage_durations_batch(sacc_array_data, d_data):
    """Build numerically safe stage durations for a padded batch of trials.

    Parameters
    ----------
    sacc_array_data : jax.Array, shape (n_trials, max_d)
        Padded stage onset times for a batch of aDDM trials.
    d_data : jax.Array, shape (n_trials,)
        Number of valid stages in each trial.

    Returns
    -------
    valid_stage_mask_data : jax.Array, shape (n_trials, max_d - 1)
        Boolean mask marking which entries of ``diff(sacc_array_data, axis=1)``
        correspond to real stage-to-stage durations.
    safe_stage_duration_array_data : jax.Array, shape (n_trials, max_d - 1)
        Per-transition duration matrix used by the batched JAX kernels. Valid
        durations are passed through unchanged. Padded transitions are replaced
        by ``_DUMMY_STAGE_DURATION`` so traced JAX code never sees padded zero
        or negative durations from the tail.

    Notes
    -----
    This is the batched analogue of ``jax.multi_stage._safe_stage_durations``.

    Example
    -------
    If

    ``sacc_array_data = [[0, 1, 3, 7, 0], [0, 0.5, 0, 0, 0]]``

    and

    ``d_data = [4, 2]``,

    then the real durations are ``[1, 2, 4]`` for the first trial and
    ``[0.5]`` for the second trial, giving

    ``valid_stage_mask_data = [[True, True, True, False], [True, False, False, False]]``

    and

    ``safe_stage_duration_array_data = [[1, 2, 4, dummy], [0.5, dummy, dummy, dummy]]``.
    """
    batch_size, max_d = sacc_array_data.shape
    dtype = sacc_array_data.dtype
    if max_d <= 1:
        return (
            jnp.empty((batch_size, 0), dtype=bool),
            jnp.empty((batch_size, 0), dtype=dtype),
        )

    raw_stage_durations = jnp.diff(sacc_array_data, axis=1)
    stage_idx = jnp.arange(max_d - 1)[None, :]
    valid_stage_mask_data = stage_idx < (d_data[:, None] - 1)
    safe_stage_duration_array_data = jnp.where(
        valid_stage_mask_data,
        raw_stage_durations,
        _DUMMY_STAGE_DURATION,
    )
    return valid_stage_mask_data, safe_stage_duration_array_data


def _effective_addm_batch_schedule(sacc_array_data, d_data, a, b):
    """Construct the batched aDDM boundary schedule.

    Parameters
    ----------
    sacc_array_data : jax.Array, shape (n_trials, max_d)
        Padded stage onset times for each trial.
    d_data : jax.Array, shape (n_trials,)
        Number of valid stages in each trial.
    a : float
        Initial upper-boundary intercept. The lower boundary starts at ``-a``.
    b : float
        Symmetric boundary-collapse slope magnitude. The upper boundary slope is
        ``-b`` and the lower boundary slope is ``+b`` on active stages.

    Returns
    -------
    safe_stage_duration_array_data : jax.Array, shape (n_trials, max_d - 1)
        Safe per-stage durations from :func:`_safe_stage_durations_batch`.
    upper_slope_array_data : jax.Array, shape (n_trials, max_d - 1)
        Upper-boundary slope for each transition. Active entries are ``-b`` and
        padded entries are ``0``.
    lower_slope_array_data : jax.Array, shape (n_trials, max_d - 1)
        Lower-boundary slope for each transition. Active entries are ``+b`` and
        padded entries are ``0``.
    a_starts_data : jax.Array, shape (n_trials, max_d)
        Upper-boundary value at the start of each stage.

    Notes
    -----
    This is the batched analogue of
    ``jax.multi_stage._effective_addm_schedule``. The padded tail is made
    inert by combining dummy positive durations with zero slopes.
    """
    batch_size, max_d = sacc_array_data.shape
    dtype = sacc_array_data.dtype
    if max_d <= 1:
        empty = jnp.empty((batch_size, 0), dtype=dtype)
        starts = jnp.full((batch_size, 1), a, dtype=dtype)
        return empty, empty, empty, starts

    valid_stage_mask_data, safe_stage_duration_array_data = _safe_stage_durations_batch(
        sacc_array_data, d_data
    )
    upper_slope_array_data = jnp.where(valid_stage_mask_data, -b, 0.0)
    lower_slope_array_data = jnp.where(valid_stage_mask_data, b, 0.0)
    a_starts_data = jnp.concatenate(
        [
            jnp.full((batch_size, 1), a, dtype=dtype),
            a
            + jnp.cumsum(
                upper_slope_array_data * safe_stage_duration_array_data, axis=1
            ),
        ],
        axis=1,
    )
    return (
        safe_stage_duration_array_data,
        upper_slope_array_data,
        lower_slope_array_data,
        a_starts_data,
    )


# ---------------------------------------------------------------------------
# Batch likelihood kernels
# ---------------------------------------------------------------------------


def compute_addm_loglikelihoods_batchvmap(
    rt_data,
    choice_data,
    eta,
    kappa,
    sigma,
    a,
    b,
    x0,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Legacy batch baseline: vmap over the scalar ADDM log-kernel.

    Notes
    -----
    This is the simplest batch implementation: it vmaps the public
    single-trial production kernel ``compute_addm_logfptd`` over trials. Since the
    single-trial production kernel in ``jax.multi_stage`` already precomputes
    stage transition matrices per trial, this path should be thought of as
    "batch by vmapping the single-trial precomputed kernel".

    Within the current JAX organization, this function is the explicit legacy
    batch baseline. The explicit single-trial stage-scan kernels live in
    ``jax.multi_stage`` under ``compute_*_stagescan``.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is
    no ``adaptive_stopping`` or ``threshold`` option in this backend.

    If ``use_remat=True``, rematerialize the vmapped single-trial production
    kernel to trade extra compute for lower reverse-mode memory use.
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )

    def single_trial_loglikelihood(rt, choice, r1, r2, flag, sacc_array, d):
        return compute_addm_logfptd(
            rt,
            choice,
            eta,
            kappa,
            sigma,
            a,
            b,
            x0,
            r1,
            r2,
            flag,
            sacc_array,
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    if use_remat:
        single_trial_loglikelihood = remat(single_trial_loglikelihood)

    return vmap(single_trial_loglikelihood, in_axes=(0, 0, 0, 0, 0, 0, 0))(
        rt_data, choice_data, r1_data, r2_data, flag_data, sacc_array_data, d_data
    )


def _compute_addm_loglikelihoods_batchscan_core(
    rt_data,
    choice_data,
    mu_array_data,
    sacc_array_data,
    d_data,
    sigma,
    a,
    b,
    x0,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Compute batched aDDM log-likelihoods with a dedicated stage-scan kernel.

    Parameters
    ----------
    rt_data, choice_data : jax.Array, shape (n_trials,)
        Observed reaction times and choices.
    mu_array_data : jax.Array, shape (n_trials, max_d)
        Per-trial, per-stage drift arrays already derived from the aDDM
        covariates.
    sacc_array_data : jax.Array, shape (n_trials, max_d)
        Padded stage onset times for each trial.
    d_data : jax.Array, shape (n_trials,)
        Number of valid stages in each trial.
    sigma, a, b, x0 : float
        Shared aDDM parameters for the batch.
    order_mid, order_last, trunc_num : int, optional
        Intermediate-stage quadrature order, final-stage quadrature order, and
        fixed single-stage truncation length.
    log_space : bool, optional
        Whether to propagate the alive-state mass in log space.
    use_remat : bool, optional
        If True, rematerialize the scan body to trade extra compute for lower
        reverse-mode memory use.

    Returns
    -------
    jax.Array, shape (n_trials,)
        Per-trial log-likelihoods.

    Notes
    -----
    This is the main batched runtime kernel. It replaces the older
    ``vmap(compute_addm_logfptd)`` baseline with a dedicated stage-wise batched
    scan that carries all trials together through the same quadrature updates.

    Unlike the single-trial production kernel in ``jax.multi_stage``, this
    function does *not* precompute all transition matrices for all stages and
    all trials up front. Instead, it computes each stage's batched transition
    matrix inside the scan body. That choice keeps memory usage and XLA temp
    storage lower than a hypothetical fully precomputed batch-transition
    tensor.

    So within the JAX backend, the four main internal variants are:

    - ``jax.multi_stage.compute_*_precomputed``: single-trial production,
      precompute all stage transitions for one trial
    - ``jax.multi_stage.compute_*_stagescan``: single-trial stage-scan,
      compute transitions inside ``lax.scan``
    - ``compute_addm_loglikelihoods_batchvmap``: legacy batch baseline, vmap the
      single-trial production path over trials
    - ``compute_addm_loglikelihoods_batchscan``: production batch kernel, compute
      batched transitions on the fly inside one shared scan
    """
    x_ref_mid, w_ref_mid = get_gauss_legendre_ref(order_mid)
    x_ref_last, w_ref_last = get_gauss_legendre_ref(order_last)

    batch_size, max_d = mu_array_data.shape
    # Handle d == 1 trials directly through the single-stage kernel so the
    # multistage recurrence only needs to serve the true multistage cases.
    upper_single = log_fptd_single(
        rt_data,
        mu_array_data[:, 0],
        sigma,
        a,
        -b,
        -a,
        b,
        x0,
        1,
        trunc_num=trunc_num,
    )
    lower_single = log_fptd_single(
        rt_data,
        mu_array_data[:, 0],
        sigma,
        a,
        -b,
        -a,
        b,
        x0,
        -1,
        trunc_num=trunc_num,
    )
    single_stage = jnp.where(choice_data == 1, upper_single, lower_single)
    if max_d < 2:
        return single_stage

    # Build per-trial stage schedules once, then organize the batched recurrence
    # into explicit first/middle/last-stage steps.
    (
        safe_stage_duration_array_data,
        upper_slope_array_data,
        lower_slope_array_data,
        a_starts_data,
    ) = _effective_addm_batch_schedule(sacc_array_data, d_data, a, b)

    def first_stage_batch():
        a_1 = a_starts_data[:, 1]
        xs_init_mid, ws_init_mid = _symmetric_stage_grid(x_ref_mid, w_ref_mid, a_1)
        xs_init_last, ws_init_last = _symmetric_stage_grid(x_ref_last, w_ref_last, a_1)
        ws_pv_init_mid = _batch_addm_first_stage_to_grid(
            xs_init_mid,
            ws_init_mid,
            mu0_data=mu_array_data[:, 0],
            sigma=sigma,
            a=a,
            upper_slope0_data=upper_slope_array_data[:, 0],
            lower_slope0_data=lower_slope_array_data[:, 0],
            stage_duration0_data=safe_stage_duration_array_data[:, 0],
            x0=x0,
            trunc_num=trunc_num,
            log_space=log_space,
        )
        ws_pv_init_last = _batch_addm_first_stage_to_grid(
            xs_init_last,
            ws_init_last,
            mu0_data=mu_array_data[:, 0],
            sigma=sigma,
            a=a,
            upper_slope0_data=upper_slope_array_data[:, 0],
            lower_slope0_data=lower_slope_array_data[:, 0],
            stage_duration0_data=safe_stage_duration_array_data[:, 0],
            x0=x0,
            trunc_num=trunc_num,
            log_space=log_space,
        )
        return xs_init_mid, ws_pv_init_mid, ws_pv_init_last

    def middle_stages_batch(carry):
        if log_space:

            def stage_step(carry, stage_idx):
                xs_prev, log_ws_pv_prev = carry
                a_prev = a_starts_data[:, stage_idx]
                a_curr = a_starts_data[:, stage_idx + 1]
                xs = x_ref_mid[None, :] * a_curr[:, None]
                ws = w_ref_mid[None, :] * a_curr[:, None]

                P = vmap(
                    lambda xs_row, mu, a_prev_val, upper_slope, lower_slope, dt, xs_prev_row: (  # noqa: E501
                        q_single(
                            xs_row[:, None],
                            mu,
                            sigma,
                            a_prev_val,
                            upper_slope,
                            -a_prev_val,
                            lower_slope,
                            dt,
                            xs_prev_row[None, :],
                            trunc_num=trunc_num,
                        )
                    )
                )(
                    xs,
                    mu_array_data[:, stage_idx],
                    a_prev,
                    upper_slope_array_data[:, stage_idx],
                    lower_slope_array_data[:, stage_idx],
                    safe_stage_duration_array_data[:, stage_idx],
                    xs_prev,
                )

                log_pv_new = logsumexp(
                    positive_log(P) + log_ws_pv_prev[:, None, :], axis=2
                )
                log_ws_pv_new = positive_log(ws) + log_pv_new
                active = stage_idx < (d_data - 2)
                xs_out = jnp.where(active[:, None], xs, xs_prev)
                log_ws_pv_out = jnp.where(
                    active[:, None], log_ws_pv_new, log_ws_pv_prev
                )
                return (xs_out, log_ws_pv_out), None

        else:

            def stage_step(carry, stage_idx):
                xs_prev, ws_pv_prev = carry
                a_prev = a_starts_data[:, stage_idx]
                a_curr = a_starts_data[:, stage_idx + 1]
                xs = x_ref_mid[None, :] * a_curr[:, None]
                ws = w_ref_mid[None, :] * a_curr[:, None]

                P = vmap(
                    lambda xs_row, mu, a_prev_val, upper_slope, lower_slope, dt, xs_prev_row: (  # noqa: E501
                        q_single(
                            xs_row[:, None],
                            mu,
                            sigma,
                            a_prev_val,
                            upper_slope,
                            -a_prev_val,
                            lower_slope,
                            dt,
                            xs_prev_row[None, :],
                            trunc_num=trunc_num,
                        )
                    )
                )(
                    xs,
                    mu_array_data[:, stage_idx],
                    a_prev,
                    upper_slope_array_data[:, stage_idx],
                    lower_slope_array_data[:, stage_idx],
                    safe_stage_duration_array_data[:, stage_idx],
                    xs_prev,
                )

                pv_new = jnp.matmul(P, ws_pv_prev[..., None]).squeeze(axis=-1)
                ws_pv_new = ws * pv_new
                active = stage_idx < (d_data - 2)
                xs_out = jnp.where(active[:, None], xs, xs_prev)
                ws_pv_out = jnp.where(active[:, None], ws_pv_new, ws_pv_prev)
                return (xs_out, ws_pv_out), None

        if use_remat:
            stage_step = remat(stage_step)

        if max_d > 3:
            carry, _ = lax.scan(stage_step, carry, jnp.arange(1, max_d - 2))
        return carry

    def last_stage_batch(xs_mid_final, ws_pv_mid_final, ws_pv_init_last):
        safe_d_idx = jnp.minimum(d_data - 1, max_d - 1)
        sacc_final = jnp.take_along_axis(
            sacc_array_data, safe_d_idx[:, None], axis=1
        ).squeeze(axis=1)
        a_final = jnp.take_along_axis(
            a_starts_data, safe_d_idx[:, None], axis=1
        ).squeeze(axis=1)
        mu_final = jnp.take_along_axis(
            mu_array_data, safe_d_idx[:, None], axis=1
        ).squeeze(axis=1)
        t_in_final_stage = rt_data - sacc_final
        xs_final, ws_final = _symmetric_stage_grid(x_ref_last, w_ref_last, a_final)

        last_q_idx = jnp.maximum(safe_d_idx - 1, 0)
        mu_last_q = jnp.take_along_axis(
            mu_array_data, last_q_idx[:, None], axis=1
        ).squeeze(axis=1)
        a_prev_last = jnp.take_along_axis(
            a_starts_data, last_q_idx[:, None], axis=1
        ).squeeze(axis=1)
        upper_last_q = jnp.take_along_axis(
            upper_slope_array_data, last_q_idx[:, None], axis=1
        ).squeeze(axis=1)
        lower_last_q = jnp.take_along_axis(
            lower_slope_array_data, last_q_idx[:, None], axis=1
        ).squeeze(axis=1)
        dt_last_q = jnp.take_along_axis(
            safe_stage_duration_array_data, last_q_idx[:, None], axis=1
        ).squeeze(axis=1)

        P_last = vmap(
            lambda xs_last_row, mu, a_prev_val, upper_slope, lower_slope, dt, xs_prev_row: (  # noqa: E501
                q_single(
                    xs_last_row[:, None],
                    mu,
                    sigma,
                    a_prev_val,
                    upper_slope,
                    -a_prev_val,
                    lower_slope,
                    dt,
                    xs_prev_row[None, :],
                    trunc_num=trunc_num,
                )
            )
        )(
            xs_final,
            mu_last_q,
            a_prev_last,
            upper_last_q,
            lower_last_q,
            dt_last_q,
            xs_mid_final,
        )

        if log_space:
            ws_pv_last = positive_log(ws_final) + logsumexp(
                positive_log(P_last) + ws_pv_mid_final[:, None, :], axis=2
            )
        else:
            ws_pv_last = ws_final * jnp.matmul(
                P_last, ws_pv_mid_final[..., None]
            ).squeeze(axis=-1)

        ws_pv_final = jnp.where((d_data > 2)[:, None], ws_pv_last, ws_pv_init_last)

        upper = fptd_single(
            t_in_final_stage[:, None],
            mu_final[:, None],
            sigma,
            a_final[:, None],
            -b,
            -a_final[:, None],
            b,
            xs_final,
            1,
            trunc_num=trunc_num,
        )
        lower = fptd_single(
            t_in_final_stage[:, None],
            mu_final[:, None],
            sigma,
            a_final[:, None],
            -b,
            -a_final[:, None],
            b,
            xs_final,
            -1,
            trunc_num=trunc_num,
        )
        fptds = jnp.where(choice_data[:, None] == 1, upper, lower)
        return _reduce_final_stage_fptds(fptds, ws_pv_final, log_space, axis=1)

    xs_init_mid, ws_pv_init_mid, ws_pv_init_last = first_stage_batch()
    xs_mid_final, ws_pv_mid_final = middle_stages_batch((xs_init_mid, ws_pv_init_mid))
    multi_stage = last_stage_batch(xs_mid_final, ws_pv_mid_final, ws_pv_init_last)

    return jnp.where(d_data == 1, single_stage, multi_stage)


def compute_addm_loglikelihoods_batchscan(
    rt_data,
    choice_data,
    eta,
    kappa,
    sigma,
    a,
    b,
    x0,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Compute ADDM log-likelihoods with the dedicated batch stage-scan kernel.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is
    no ``adaptive_stopping`` or ``threshold`` option in this backend.

    This public wrapper keeps the high-level addm signature
    ``(eta, kappa, r1_data, r2_data, flag_data)`` and dispatches to the
    explicit batch stage-scan kernel after building the derived drifts.

    If ``use_remat=True``, rematerialize the batch scan body to trade extra
    compute for lower reverse-mode memory use.
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )
    mu_array_data = _build_addm_mu_array_data(
        eta, kappa, r1_data, r2_data, flag_data, d_data, sacc_array_data.shape[1]
    )
    return _compute_addm_loglikelihoods_batchscan_core(
        rt_data,
        choice_data,
        mu_array_data,
        sacc_array_data,
        d_data,
        sigma,
        a,
        b,
        x0,
        order_mid=order_mid,
        order_last=order_last,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
    )


# ---------------------------------------------------------------------------
# Public batch reductions
# ---------------------------------------------------------------------------


def compute_addm_nll(
    rt_data,
    choice_data,
    eta,
    kappa,
    sigma,
    a,
    b,
    x0,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
    reduce="mean",
    invalid_policy="inf",
    warn=True,
):
    """Compute negative log-likelihood for a batch of addm trials.

    Parameters
    ----------
    order_mid, order_last, trunc_num : int, optional
        Intermediate-stage quadrature order, final-stage quadrature order, and
        fixed single-stage truncation length. JAX does not expose
        ``adaptive_stopping`` or ``threshold`` in this backend.
    use_remat : bool, optional
        If True, rematerialize the selected batch log-likelihood kernel to trade
        extra compute for lower reverse-mode memory use.
    reduce : str, optional
        ``"mean"`` (default) or ``"sum"``.
    invalid_policy : {"inf", "warn"}, optional
        ``"inf"`` propagates bad trials to ``+inf`` or ``NaN``. ``"warn"``
        warns and skips them.
    warn : bool, optional
        If True, emit warnings for bad trials. Defaults to True for this
        one-shot convenience API. For hot optimization loops, prefer
        :func:`make_addm_nll_function`, whose returned jitted closure stays
        silent by default.
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )
    loglikelihoods = compute_addm_loglikelihoods(
        rt_data,
        choice_data,
        eta,
        kappa,
        sigma,
        a,
        b,
        x0,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        order_mid=order_mid,
        order_last=order_last,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
    )
    return _reduce_addm_loglikelihoods_to_nll(
        loglikelihoods,
        reduce=reduce,
        invalid_policy=invalid_policy,
        warn=warn,
    )


# ---------------------------------------------------------------------------
# NLL closure builders
# ---------------------------------------------------------------------------


def _build_addm_nll_function_with_kernel(
    kernel,
    rt_data,
    choice_data,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
    invalid_policy="inf",
):
    """Build a jitted parameter-only NLL closure from a log-likelihood kernel.

    The returned function closes over fixed data and exposes only the model
    parameters ``(eta, kappa, sigma, a, b, x0)``.

    This closure never emits Python warnings from inside the jitted path.
    Call :func:`compute_addm_nll` directly when warning diagnostics for bad
    trials are desired.
    """

    @jit
    def nll_fn(eta, kappa, sigma, a, b, x0):
        loglikelihoods = kernel(
            rt_data,
            choice_data,
            eta,
            kappa,
            sigma,
            a,
            b,
            x0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=log_space,
            use_remat=use_remat,
        )
        return _reduce_addm_loglikelihoods_to_nll(
            loglikelihoods,
            reduce="sum",
            invalid_policy=invalid_policy,
            warn=False,
        )

    return nll_fn


def make_addm_nll_function_batchvmap(
    rt_data,
    choice_data,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
    invalid_policy="inf",
):
    """Build an optimization closure using the legacy vmap baseline kernel.

    If ``use_remat=True``, rematerialize the vmapped single-trial production
    kernel to trade extra compute for lower reverse-mode memory use.
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )

    def kernel(
        rt_data,
        choice_data,
        eta,
        kappa,
        sigma,
        a,
        b,
        x0,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        *,
        order_mid=order_mid,
        order_last=order_last,
        trunc_num=DEFAULT_TRUNC_NUM,
        log_space=False,
        use_remat=False,
    ):
        return compute_addm_loglikelihoods_batchvmap(
            rt_data,
            choice_data,
            eta,
            kappa,
            sigma,
            a,
            b,
            x0,
            r1_data,
            r2_data,
            flag_data,
            sacc_array_data,
            d_data,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=log_space,
            use_remat=use_remat,
        )

    return _build_addm_nll_function_with_kernel(
        kernel,
        rt_data,
        choice_data,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        order_mid=order_mid,
        order_last=order_last,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
        invalid_policy=invalid_policy,
    )


def make_addm_nll_function_batchscan(
    rt_data,
    choice_data,
    r1_data,
    r2_data,
    flag_data,
    sacc_array_data,
    d_data,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
    invalid_policy="inf",
):
    """Create a batched ADDM negative log-likelihood function for optimization.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is
    no ``adaptive_stopping`` or ``threshold`` option in this backend.

    If ``use_remat=True``, rematerialize the batch stage-scan body to trade
    extra compute for lower reverse-mode memory use.
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )

    def kernel(
        rt_data,
        choice_data,
        eta,
        kappa,
        sigma,
        a,
        b,
        x0,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        *,
        order_mid=order_mid,
        order_last=order_last,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
    ):
        mu_array_data = _build_addm_mu_array_data(
            eta, kappa, r1_data, r2_data, flag_data, d_data, sacc_array_data.shape[1]
        )
        return _compute_addm_loglikelihoods_batchscan_core(
            rt_data,
            choice_data,
            mu_array_data,
            sacc_array_data,
            d_data,
            sigma,
            a,
            b,
            x0,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=log_space,
            use_remat=use_remat,
        )

    return _build_addm_nll_function_with_kernel(
        kernel,
        rt_data,
        choice_data,
        r1_data,
        r2_data,
        flag_data,
        sacc_array_data,
        d_data,
        order_mid=order_mid,
        order_last=order_last,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
        invalid_policy=invalid_policy,
    )


# ---------------------------------------------------------------------------
# Public aliases and JIT wrappers
# ---------------------------------------------------------------------------


compute_addm_loglikelihoods = compute_addm_loglikelihoods_batchscan
make_addm_nll_function = make_addm_nll_function_batchscan
compute_addm_loglikelihoods_jit = jit(
    compute_addm_loglikelihoods,
    static_argnames=(
        "order_mid",
        "order_last",
        "order",
        "trunc_num",
        "log_space",
        "use_remat",
    ),
)


# ---------------------------------------------------------------------------
# HSSM-authored: public wrapper over the private batchscan core.
# ---------------------------------------------------------------------------
# This is the only HSSM-authored logic addition to the vendored tree (preserve
# on re-vendor). It exposes a stable public entry that accepts a *pre-built*
# per-trial drift array ``mu_array`` directly, so a custom attention process can
# supply its own drift schedule instead of having the kernel build the default
# alternating array internally (as ``compute_addm_loglikelihoods`` does).
def compute_addm_loglikelihoods_from_mu(
    rt,
    choice,
    mu_array,
    sacc_array,
    d,
    sigma,
    a,
    b,
    x0,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
    use_remat=False,
):
    """Batched aDDM log-likelihoods from a pre-built drift array.

    Thin public alias for :func:`_compute_addm_loglikelihoods_batchscan_core`.
    Unlike :func:`compute_addm_loglikelihoods` (which builds the per-trial drift
    array internally from ``eta, kappa, r1, r2, flag``), this entry accepts the
    drift array ``mu_array`` directly, giving custom attention processes a stable
    seam onto the same kernel.

    Parameters
    ----------
    rt, choice : jax.Array, shape (n_trials,)
        Observed reaction times and choices.
    mu_array : jax.Array, shape (n_trials, max_d)
        Per-trial, per-stage drift array.
    sacc_array : jax.Array, shape (n_trials, max_d)
        Padded stage onset times for each trial.
    d : jax.Array, shape (n_trials,)
        Number of valid stages in each trial.
    sigma, a, b, x0 : float
        Shared aDDM parameters for the batch.
    order_mid, order_last, trunc_num : int, optional
        Intermediate-stage quadrature order, final-stage quadrature order, and
        fixed single-stage truncation length.
    log_space : bool, optional
        Whether to propagate the alive-state mass in log space.
    use_remat : bool, optional
        If True, rematerialize the scan body to trade extra compute for lower
        reverse-mode memory use.

    Returns
    -------
    jax.Array, shape (n_trials,)
        Per-trial log-likelihoods.
    """
    return _compute_addm_loglikelihoods_batchscan_core(
        rt,
        choice,
        mu_array,
        sacc_array,
        d,
        sigma,
        a,
        b,
        x0,
        order_mid=order_mid,
        order_last=order_last,
        trunc_num=trunc_num,
        log_space=log_space,
        use_remat=use_remat,
    )
