# Vendored from efficient-fpt @ d97a451; do not edit in place — re-vendor instead.
# HSSM-authored change to the vendored copy: above-package imports retargeted to
# siblings (``._defaults``, ``.utils``). Preserve on re-vendor.
"""JAX single-trial multi-stage first-passage time density computation.

This module holds *both* explicit single-trial JAX implementations:

- public production kernels:
  ``compute_addm_logfptd_precomputed`` and
  ``compute_heterog_multistage_logfptd_precomputed``
- public stage-scan kernels:
  ``compute_addm_logfptd_stagescan`` and
  ``compute_heterog_multistage_logfptd_stagescan``

The production path precomputes all stage transition matrices for one trial
with ``vmap`` over stage index, then propagates the alive-state mass through
those transitions. The stage-scan path computes transition matrices inside
``lax.scan`` as the stage recurrence advances. That path is slower, but it is
useful for correctness checks, benchmarking, and algorithm comparison.

Relationship to the other JAX modules
-------------------------------------
- ``jax.batch`` contains the batched aDDM kernels. Its legacy baseline vmaps
  the public single-trial production function in this module over trials,
  while its production batch kernel carries the whole batch through one
  dedicated stage scan and computes transitions on the fly stage by stage.

So the split is:
- this file: single-trial precomputed + single-trial stage-scan
- ``jax.batch``: batch legacy baseline + batch production

The plain aliases ``compute_addm_logfptd`` and
``compute_heterog_multistage_logfptd`` continue to point to the precomputed
production kernels.

JAX uses a fixed-length series controlled only by ``trunc_num``. There is no
``adaptive_stopping`` or ``threshold`` option in this backend.
"""

import jax.numpy as jnp
from jax import jit, lax, vmap
from jax.scipy.special import logsumexp

from .addm_helpers import _build_addm_mu_array
from .single_stage import fptd_single, q_single, log_fptd_single
from .utils import get_gauss_legendre_ref, _DUMMY_STAGE_DURATION, positive_log
from ._defaults import (
    DEFAULT_LAST_QUAD_ORDER,
    DEFAULT_MID_QUAD_ORDER,
    DEFAULT_TRUNC_NUM,
)
from .utils import resolve_quadrature_orders


# ---------------------------------------------------------------------------
# Tiny numeric helpers
# ---------------------------------------------------------------------------

def _safe_stage_durations(node_array, d):
    """Build a numerically safe duration vector from a padded stage-time array.

    Parameters
    ----------
    node_array : jax.Array, shape (max_d,)
        Padded stage start times. Only the first ``d`` entries are treated as
        valid stage onsets; later entries may be arbitrary padding.
    d : int
        Number of valid stages.

    Returns
    -------
    valid_stage_mask : jax.Array, shape (max_d - 1,)
        Boolean mask marking which entries of ``diff(node_array)`` correspond to
        real stage-to-stage durations. For ``d`` valid stages there are exactly
        ``d - 1`` valid transitions.
    safe_stage_duration_array : jax.Array, shape (max_d - 1,)
        Per-transition duration array used by the JAX multistage kernels.
        Valid durations are passed through unchanged. Inactive padded
        transitions are replaced by ``_DUMMY_STAGE_DURATION`` so traced JAX code
        never sees padded zero or negative stage lengths.

    Notes
    -----
    This helper exists because the JAX kernels operate on fixed-size padded
    arrays. Even inactive padded stages still participate in tracing, so they
    must be given harmless positive durations.

    Example
    -------
    If ``node_array = [0, 1, 3, 7, 0, 0, 0]`` and ``d = 4``, then the valid
    stages start at times ``0, 1, 3, 7``. The real stage durations are
    ``[1, 2, 4]`` and the padded tail is ignored:

    ``valid_stage_mask = [True, True, True, False, False, False]``

    ``safe_stage_duration_array = [1, 2, 4, dummy, dummy, dummy]``
    """
    max_d = node_array.shape[0]
    dtype = node_array.dtype
    if max_d <= 1:
        empty = jnp.empty((0,), dtype=dtype)
        return empty, empty

    raw_stage_durations = jnp.diff(node_array)
    stage_idx = jnp.arange(max_d - 1)
    valid_stage_mask = stage_idx < (d - 1)
    safe_stage_duration_array = jnp.where(
        valid_stage_mask,
        raw_stage_durations,
        _DUMMY_STAGE_DURATION,
    )
    return valid_stage_mask, safe_stage_duration_array


# ---------------------------------------------------------------------------
# Schedule builders
# ---------------------------------------------------------------------------


def _effective_addm_schedule(sacc_array, d, a, b):
    """Construct the stage schedule for the symmetric aDDM boundary geometry.

    Parameters
    ----------
    sacc_array : jax.Array, shape (max_d,)
        Padded aDDM stage onset times.
    d : int
        Number of valid stages.
    a : float
        Initial upper-boundary intercept. The lower boundary starts at ``-a``.
    b : float
        Symmetric boundary-collapse slope magnitude. The upper boundary slope is
        ``-b`` and the lower boundary slope is ``+b`` on active stages.

    Returns
    -------
    safe_stage_duration_array : jax.Array, shape (max_d - 1,)
        Safe per-stage durations from :func:`_safe_stage_durations`.
    upper_slope_array : jax.Array, shape (max_d - 1,)
        Upper-boundary slope per transition. Active entries are ``-b`` and
        inactive padded entries are ``0``.
    lower_slope_array : jax.Array, shape (max_d - 1,)
        Lower-boundary slope per transition. Active entries are ``+b`` and
        inactive padded entries are ``0``.
    a_starts : jax.Array, shape (max_d,)
        Upper-boundary value at the start of each stage. Because the aDDM
        boundary is symmetric, the lower-boundary start is always ``-a_starts``.

    Notes
    -----
    The multistage aDDM kernel uses ``a_starts[k]`` to place quadrature nodes at
    the start of stage ``k`` and uses ``upper_slope_array[k]`` / ``lower_slope_array[k]``
    to propagate the boundaries through the corresponding duration.

    Example
    -------
    If ``sacc_array = [0, 1, 3, 7, 0, 0, 0]``, ``d = 4``, ``a = 1.5``, and
    ``b = 0.3``, then:

    ``safe_stage_duration_array = [1, 2, 4, dummy, dummy, dummy]``

    ``upper_slope_array = [-0.3, -0.3, -0.3, 0, 0, 0]``

    ``lower_slope_array = [0.3, 0.3, 0.3, 0, 0, 0]``

    ``a_starts = [1.5, 1.2, 0.6, -0.6, -0.6, -0.6, -0.6]``

    The repeated tail means the padded stages are inert: once the valid stages
    end, the boundary start stays frozen.
    """
    dtype = sacc_array.dtype
    valid_stage_mask, safe_stage_duration_array = _safe_stage_durations(sacc_array, d)
    upper_slope_array = jnp.where(valid_stage_mask, -b, 0.0)
    lower_slope_array = jnp.where(valid_stage_mask, b, 0.0)
    a_starts = jnp.concatenate(
        [
            jnp.full((1,), a, dtype=dtype),
            a + jnp.cumsum(upper_slope_array * safe_stage_duration_array),
        ]
    )
    return safe_stage_duration_array, upper_slope_array, lower_slope_array, a_starts


def _effective_general_schedule(node_array, d, a1, b1_array, a2, b2_array):
    """Construct the stage schedule for the general asymmetric multistage model.

    Parameters
    ----------
    node_array : jax.Array, shape (max_d,)
        Padded stage onset times.
    d : int
        Number of valid stages.
    a1, a2 : float
        Upper and lower boundary intercepts at the start of stage 0.
    b1_array, b2_array : jax.Array, shape (max_d,)
        Per-stage upper and lower boundary slopes. Only the first ``d`` entries
        are meaningful.

    Returns
    -------
    safe_stage_duration_array : jax.Array, shape (max_d - 1,)
        Safe per-stage durations from :func:`_safe_stage_durations`.
    upper_slope_array : jax.Array, shape (max_d - 1,)
        Active upper-boundary slopes copied from ``b1_array[:-1]``; padded
        entries are set to ``0``.
    lower_slope_array : jax.Array, shape (max_d - 1,)
        Active lower-boundary slopes copied from ``b2_array[:-1]``; padded
        entries are set to ``0``.
    ub_starts : jax.Array, shape (max_d,)
        Upper boundary value at the start of each stage.
    lb_starts : jax.Array, shape (max_d,)
        Lower boundary value at the start of each stage.

    Notes
    -----
    This is the generalized analogue of :func:`_effective_addm_schedule`.
    Unlike the aDDM helper, upper and lower boundaries are tracked separately.

    Example
    -------
    If ``node_array = [0, 1, 3, 7, 0, 0, 0]``, ``d = 4``,
    ``a1 = 1.5``, ``a2 = -1.5``,
    ``b1_array = [-0.3, -0.1, 0.0, 0, 0, 0, 0]``, and
    ``b2_array = [0.2, 0.4, 0.1, 0, 0, 0, 0]``, then:

    ``safe_stage_duration_array = [1, 2, 4, dummy, dummy, dummy]``

    ``upper_slope_array = [-0.3, -0.1, 0.0, 0, 0, 0]``

    ``lower_slope_array = [0.2, 0.4, 0.1, 0, 0, 0]``

    ``ub_starts = [1.5, 1.2, 1.0, 1.0, 1.0, 1.0, 1.0]``

    ``lb_starts = [-1.5, -1.3, -0.5, -0.1, -0.1, -0.1, -0.1]``
    """
    dtype = node_array.dtype
    valid_stage_mask, safe_stage_duration_array = _safe_stage_durations(node_array, d)
    upper_slope_array = jnp.where(valid_stage_mask, b1_array[:-1], 0.0)
    lower_slope_array = jnp.where(valid_stage_mask, b2_array[:-1], 0.0)
    ub_starts = jnp.concatenate(
        [
            jnp.full((1,), a1, dtype=dtype),
            a1 + jnp.cumsum(upper_slope_array * safe_stage_duration_array),
        ]
    )
    lb_starts = jnp.concatenate(
        [
            jnp.full((1,), a2, dtype=dtype),
            a2 + jnp.cumsum(lower_slope_array * safe_stage_duration_array),
        ]
    )
    return (
        safe_stage_duration_array,
        upper_slope_array,
        lower_slope_array,
        ub_starts,
        lb_starts,
    )


# ---------------------------------------------------------------------------
# Shared precomputed-transition propagation helper
# ---------------------------------------------------------------------------


def _propagate_ws_pv(P_all, ws_all, ws_pv_init, num_active, log_space):
    """Propagate ws_pv across precomputed stage transitions.
    ws_pv = ws * pv (elementwise product),
    where ws is the quadrature weight and pv is non-passive density.

    Parameters
    ----------
    P_all : jax.Array, shape (n_transitions, order_mid, order_mid)
        Precomputed mid-stage transition matrices between consecutive
        intermediate interfaces.
    ws_all : jax.Array, shape (order_mid, n_interfaces)
        Intermediate-interface quadrature weights.
    ws_pv_init : jax.Array, shape (order_mid,)
        Weighted alive-state mass after the first stage.
    num_active : int
        Number of active transition matrices to apply.
    log_space : bool
        Whether ``ws_pv_init`` and the propagated state are represented in log
        space.

    Returns
    -------
    jax.Array, shape (order_mid,)
        Weighted alive-state mass at the start of the final q-propagation stage.

    Notes
    -----
    The scan still iterates over the padded tail for fixed-shape JAX tracing,
    but once ``k >= num_active`` the carry is left unchanged.
    """
    stage_indices = jnp.arange(P_all.shape[0])

    if log_space:
        log_P_all = positive_log(P_all)

        def mv_step(log_ws_pv_prev, k):
            log_pv_new = logsumexp(log_P_all[k] + log_ws_pv_prev[None, :], axis=1)
            log_ws_pv_new = positive_log(ws_all[:, k + 1]) + log_pv_new
            active = k < num_active
            return jnp.where(active, log_ws_pv_new, log_ws_pv_prev), None

    else:

        def mv_step(ws_pv_prev, k):
            ws_k = ws_all[:, k + 1]
            pv_new = P_all[k] @ ws_pv_prev
            ws_pv_new = ws_k * pv_new
            active = k < num_active
            return jnp.where(active, ws_pv_new, ws_pv_prev), None

    ws_pv_final, _ = lax.scan(mv_step, ws_pv_init, stage_indices)
    return ws_pv_final


def _apply_transition(P, ws_dest, ws_pv_prev, log_space):
    """Apply one transition matrix to a weighted alive-state carry."""
    if log_space:
        log_pv = logsumexp(positive_log(P) + ws_pv_prev[None, :], axis=1)
        return positive_log(ws_dest) + log_pv
    pv = P @ ws_pv_prev
    return ws_dest * pv


def _symmetric_stage_grid(x_ref, w_ref, a_stage):
    """Map reference quadrature nodes/weights to a symmetric interval."""
    a_stage = jnp.asarray(a_stage)
    return x_ref * a_stage[..., None], w_ref * a_stage[..., None]


def _general_stage_grid(x_ref, w_ref, ub_stage, lb_stage):
    """Map reference quadrature nodes/weights to an asymmetric interval."""
    half_width = (ub_stage - lb_stage) / 2.0
    center = (ub_stage + lb_stage) / 2.0
    half_width = jnp.asarray(half_width)
    center = jnp.asarray(center)
    return (
        x_ref * half_width[..., None] + center[..., None],
        w_ref * half_width[..., None],
    )


def _reduce_final_stage_fptds(fptds, ws_pv_final, log_space, *, axis=None):
    """Reduce final-stage densities against weighted alive-state mass."""
    if log_space:
        return logsumexp(positive_log(fptds) + ws_pv_final, axis=axis)
    return positive_log(jnp.sum(fptds * ws_pv_final, axis=axis))


def _addm_first_stage_to_grid(
    xs_target,
    ws_target,
    *,
    mu0,
    sigma,
    a,
    upper_slope0,
    lower_slope0,
    stage_duration0,
    x0,
    trunc_num,
    log_space,
):
    """Initialize first-stage weighted alive-state mass on a target grid."""
    pv_init = q_single(
        xs_target,
        mu0,
        sigma,
        a,
        upper_slope0,
        -a,
        lower_slope0,
        stage_duration0,
        x0,
        trunc_num=trunc_num,
    )
    ws_pv = ws_target * pv_init
    return positive_log(ws_pv) if log_space else ws_pv


def _general_first_stage_to_grid(
    xs_target,
    ws_target,
    *,
    x0,
    mu0,
    sigma0,
    upper_start0,
    lower_start0,
    upper_slope0,
    lower_slope0,
    stage_duration0,
    trunc_num,
    log_space,
):
    """Initialize first-stage weighted alive-state mass on a target grid."""
    pv_init = q_single(
        xs_target,
        mu0,
        sigma0,
        upper_start0,
        upper_slope0,
        lower_start0,
        lower_slope0,
        stage_duration0,
        x0,
        trunc_num=trunc_num,
    )
    ws_pv = ws_target * pv_init
    return positive_log(ws_pv) if log_space else ws_pv


def _finish_addm_last_stage(
    t_in_final_stage,
    choice,
    *,
    mu_final,
    sigma,
    a_final,
    b,
    xs_final,
    ws_pv_final,
    trunc_num,
    log_space,
):
    """Evaluate and reduce the final-stage ADDM log-FPTD."""
    fptds = fptd_single(
        t_in_final_stage,
        mu_final,
        sigma,
        a_final,
        -b,
        -a_final,
        b,
        xs_final,
        choice,
        trunc_num=trunc_num,
    )
    return _reduce_final_stage_fptds(fptds, ws_pv_final, log_space)


def _finish_general_last_stage(
    t_in_final_stage,
    choice,
    *,
    mu_final,
    sigma_final,
    ub_final,
    lb_final,
    b1_final,
    b2_final,
    xs_final,
    ws_pv_final,
    trunc_num,
    log_space,
):
    """Evaluate and reduce the final-stage generalized log-FPTD."""
    fptds = fptd_single(
        t_in_final_stage,
        mu_final,
        sigma_final,
        ub_final,
        b1_final,
        lb_final,
        b2_final,
        xs_final,
        choice,
        trunc_num=trunc_num,
    )
    return _reduce_final_stage_fptds(fptds, ws_pv_final, log_space)


# ---------------------------------------------------------------------------
# Single-trial production kernels: precomputed transitions
# ---------------------------------------------------------------------------


def _addm_logfptd_precomputed(
    rt,
    choice,
    sigma,
    a,
    b,
    x0,
    mu_array,
    sacc_array,
    d,
    *,
    order_mid,
    order_last,
    trunc_num,
    log_space,
):
    """Single-trial ADDM log-FPTD with precomputed transition matrices.

    The computation skeleton is:

    1. build the per-stage boundary schedule
    2. place quadrature nodes and weights at stage interfaces
    3. propagate alive-state mass through intermediate stages
    4. evaluate final-stage boundary-hit densities
    5. reduce over latent start positions in the final stage

    Notes
    -----
    This is the production single-trial JAX kernel. For one trial, it
    precomputes the intermediate stage transition matrices with ``vmap`` over
    stage index, then propagates the alive-state quadrature mass through them.

    This should be contrasted with:

    - ``compute_addm_logfptd_stagescan`` below in this same module:
      single-trial, but computes transitions inside ``lax.scan``
    - ``jax.batch.compute_addm_loglikelihoods_batchvmap``: batch baseline that vmaps
      the public single-trial API over trials
    - ``jax.batch.compute_addm_loglikelihoods_batchscan``: production batch kernel
      that carries the whole batch through one scan and computes transitions
      on the fly stage by stage
    """
    max_d = mu_array.shape[0]
    if max_d < 2:
        return log_fptd_single(
            rt, mu_array[0], sigma, a, -b, -a, b, x0, choice, trunc_num=trunc_num
        )
    x_ref_mid, w_ref_mid = get_gauss_legendre_ref(order_mid)
    x_ref_last, w_ref_last = get_gauss_legendre_ref(order_last)

    # Build stage-local boundary geometry, then place interface quadrature
    # nodes/weights for every potential stage transition.
    safe_stage_duration_array, upper_slope_array, lower_slope_array, a_starts = (
        _effective_addm_schedule(sacc_array, d, a, b)
    )
    safe_d_idx = jnp.minimum(d - 1, max_d - 1)

    def first_stage_last(_):
        xs_final, ws_final = _symmetric_stage_grid(x_ref_last, w_ref_last, a_starts[1])
        ws_pv_final = _addm_first_stage_to_grid(
            xs_final,
            ws_final,
            mu0=mu_array[0],
            sigma=sigma,
            a=a,
            upper_slope0=upper_slope_array[0],
            lower_slope0=lower_slope_array[0],
            stage_duration0=safe_stage_duration_array[0],
            x0=x0,
            trunc_num=trunc_num,
            log_space=log_space,
        )
        return _finish_addm_last_stage(
            rt - sacc_array[1],
            choice,
            mu_final=mu_array[1],
            sigma=sigma,
            a_final=a_starts[1],
            b=b,
            xs_final=xs_final,
            ws_pv_final=ws_pv_final,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    def first_stage_mid():
        xs_mid_all = x_ref_mid[:, None] * a_starts[1:-1]
        ws_mid_all = w_ref_mid[:, None] * a_starts[1:-1]
        ws_pv_mid = _addm_first_stage_to_grid(
            xs_mid_all[:, 0],
            ws_mid_all[:, 0],
            mu0=mu_array[0],
            sigma=sigma,
            a=a,
            upper_slope0=upper_slope_array[0],
            lower_slope0=lower_slope_array[0],
            stage_duration0=safe_stage_duration_array[0],
            x0=x0,
            trunc_num=trunc_num,
            log_space=log_space,
        )
        return xs_mid_all, ws_mid_all, ws_pv_mid

    def middle_stages(xs_mid_all, ws_mid_all, ws_pv_mid):
        if max_d > 3:

            def compute_mid_transition(k):
                stage_idx = k + 1
                a_prev = a_starts[stage_idx]
                return q_single(
                    xs_mid_all[:, k + 1, None],
                    mu_array[stage_idx],
                    sigma,
                    a_prev,
                    upper_slope_array[stage_idx],
                    -a_prev,
                    lower_slope_array[stage_idx],
                    safe_stage_duration_array[stage_idx],
                    xs_mid_all[:, k, None].T,
                    trunc_num=trunc_num,
                )

            P_mid_all = vmap(compute_mid_transition)(jnp.arange(max_d - 3))
            return _propagate_ws_pv(
                P_mid_all,
                ws_mid_all,
                ws_pv_mid,
                jnp.maximum(d - 3, 0),
                log_space,
            )
        return ws_pv_mid

    def last_stage_from_mid(xs_mid_all, ws_pv_mid):
        last_q_stage_idx = safe_d_idx - 1
        source_mid_idx = jnp.maximum(last_q_stage_idx - 1, 0)
        xs_mid_source = xs_mid_all[:, source_mid_idx]
        a_prev_last = a_starts[last_q_stage_idx]
        a_final = a_starts[safe_d_idx]
        xs_final = x_ref_last * a_final
        ws_final = w_ref_last * a_final
        P_last = q_single(
            xs_final[:, None],
            mu_array[last_q_stage_idx],
            sigma,
            a_prev_last,
            upper_slope_array[last_q_stage_idx],
            -a_prev_last,
            lower_slope_array[last_q_stage_idx],
            safe_stage_duration_array[last_q_stage_idx],
            xs_mid_source[None, :],
            trunc_num=trunc_num,
        )
        ws_pv_final = _apply_transition(P_last, ws_final, ws_pv_mid, log_space)
        return _finish_addm_last_stage(
            rt - sacc_array[safe_d_idx],
            choice,
            mu_final=mu_array[safe_d_idx],
            sigma=sigma,
            a_final=a_final,
            b=b,
            xs_final=xs_final,
            ws_pv_final=ws_pv_final,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    def multi_stage_from_mid(_):
        xs_mid_all, ws_mid_all, ws_pv_mid = first_stage_mid()
        ws_pv_mid = middle_stages(xs_mid_all, ws_mid_all, ws_pv_mid)
        return last_stage_from_mid(xs_mid_all, ws_pv_mid)

    if max_d == 2:
        return first_stage_last(None)
    return lax.cond(safe_d_idx == 1, first_stage_last, multi_stage_from_mid, operand=None)


def compute_addm_logfptd_precomputed(
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
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
):
    """Multi-stage log-FPTD for attention-dependent drift diffusion model.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is no
    ``adaptive_stopping`` or ``threshold`` option in this backend.

    This public wrapper keeps the high-level addm signature
    ``(eta, kappa, r1, r2, flag)`` and dispatches to the explicit production
    kernel after building the derived stage drifts.
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )
    mu_array = _build_addm_mu_array(
        eta,
        kappa,
        r1,
        r2,
        flag,
        d,
        sacc_array.shape[0],
        sacc_array.dtype,
    )

    def single_fn(_):
        return log_fptd_single(
            rt,
            mu_array[0],
            sigma,
            a,
            -b,
            -a,
            b,
            x0,
            choice,
            trunc_num=trunc_num,
        )

    if mu_array.shape[0] < 2:
        return single_fn(None)

    def multi_fn(_):
        return _addm_logfptd_precomputed(
            rt,
            choice,
            sigma,
            a,
            b,
            x0,
            mu_array,
            sacc_array,
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    return lax.cond(d == 1, single_fn, multi_fn, operand=None)


# ---------------------------------------------------------------------------
# Generalized single-trial production kernel: precomputed transitions
# ---------------------------------------------------------------------------


def _heterog_multistage_logfptd_precomputed(
    rt,
    choice,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    d,
    *,
    order_mid,
    order_last,
    trunc_num,
    log_space,
):
    """Generalized multi-stage log-FPTD with precomputed transitions.

    The computation skeleton mirrors :func:`_addm_logfptd_precomputed`, but with
    separate upper/lower boundary schedules and per-stage sigma values.
    """
    max_d = mu_array.shape[0]
    if max_d < 2:
        return log_fptd_single(
            rt,
            mu_array[0],
            sigma_array[0],
            a1,
            b1_array[0],
            a2,
            b2_array[0],
            x0,
            choice,
            trunc_num=trunc_num,
        )
    x_ref_mid, w_ref_mid = get_gauss_legendre_ref(order_mid)
    x_ref_last, w_ref_last = get_gauss_legendre_ref(order_last)

    (
        safe_stage_duration_array,
        upper_slope_array,
        lower_slope_array,
        ub_starts,
        lb_starts,
    ) = _effective_general_schedule(node_array, d, a1, b1_array, a2, b2_array)

    safe_d_idx = jnp.minimum(d - 1, max_d - 1)
    half_w_mid = (ub_starts[1:-1] - lb_starts[1:-1]) / 2.0
    center_mid = (ub_starts[1:-1] + lb_starts[1:-1]) / 2.0

    def first_stage_last(_):
        xs_final, ws_final = _general_stage_grid(
            x_ref_last, w_ref_last, ub_starts[1], lb_starts[1]
        )
        ws_pv_final = _general_first_stage_to_grid(
            xs_final,
            ws_final,
            x0=x0,
            mu0=mu_array[0],
            sigma0=sigma_array[0],
            upper_start0=a1,
            lower_start0=a2,
            upper_slope0=upper_slope_array[0],
            lower_slope0=lower_slope_array[0],
            stage_duration0=safe_stage_duration_array[0],
            trunc_num=trunc_num,
            log_space=log_space,
        )
        return _finish_general_last_stage(
            rt - node_array[1],
            choice,
            mu_final=mu_array[1],
            sigma_final=sigma_array[1],
            ub_final=ub_starts[1],
            lb_final=lb_starts[1],
            b1_final=b1_array[1],
            b2_final=b2_array[1],
            xs_final=xs_final,
            ws_pv_final=ws_pv_final,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    def first_stage_mid():
        xs_mid_all = x_ref_mid[:, None] * half_w_mid + center_mid
        ws_mid_all = w_ref_mid[:, None] * half_w_mid
        ws_pv_mid = _general_first_stage_to_grid(
            xs_mid_all[:, 0],
            ws_mid_all[:, 0],
            x0=x0,
            mu0=mu_array[0],
            sigma0=sigma_array[0],
            upper_start0=a1,
            lower_start0=a2,
            upper_slope0=upper_slope_array[0],
            lower_slope0=lower_slope_array[0],
            stage_duration0=safe_stage_duration_array[0],
            trunc_num=trunc_num,
            log_space=log_space,
        )
        return xs_mid_all, ws_mid_all, ws_pv_mid

    def middle_stages(xs_mid_all, ws_mid_all, ws_pv_mid):
        if max_d > 3:

            def compute_mid_transition(k):
                stage_idx = k + 1
                return q_single(
                    xs_mid_all[:, k + 1, None],
                    mu_array[stage_idx],
                    sigma_array[stage_idx],
                    ub_starts[stage_idx],
                    upper_slope_array[stage_idx],
                    lb_starts[stage_idx],
                    lower_slope_array[stage_idx],
                    safe_stage_duration_array[stage_idx],
                    xs_mid_all[:, k, None].T,
                    trunc_num=trunc_num,
                )

            P_mid_all = vmap(compute_mid_transition)(jnp.arange(max_d - 3))
            return _propagate_ws_pv(
                P_mid_all,
                ws_mid_all,
                ws_pv_mid,
                jnp.maximum(d - 3, 0),
                log_space,
            )
        return ws_pv_mid

    def last_stage_from_mid(xs_mid_all, ws_pv_mid):
        last_q_stage_idx = safe_d_idx - 1
        source_mid_idx = jnp.maximum(last_q_stage_idx - 1, 0)
        xs_mid_source = xs_mid_all[:, source_mid_idx]
        half_w_last = (ub_starts[safe_d_idx] - lb_starts[safe_d_idx]) / 2.0
        center_last = (ub_starts[safe_d_idx] + lb_starts[safe_d_idx]) / 2.0
        xs_final = x_ref_last * half_w_last + center_last
        ws_final = w_ref_last * half_w_last
        P_last = q_single(
            xs_final[:, None],
            mu_array[last_q_stage_idx],
            sigma_array[last_q_stage_idx],
            ub_starts[last_q_stage_idx],
            upper_slope_array[last_q_stage_idx],
            lb_starts[last_q_stage_idx],
            lower_slope_array[last_q_stage_idx],
            safe_stage_duration_array[last_q_stage_idx],
            xs_mid_source[None, :],
            trunc_num=trunc_num,
        )
        ws_pv_final = _apply_transition(P_last, ws_final, ws_pv_mid, log_space)
        return _finish_general_last_stage(
            rt - node_array[safe_d_idx],
            choice,
            mu_final=mu_array[safe_d_idx],
            sigma_final=sigma_array[safe_d_idx],
            ub_final=ub_starts[safe_d_idx],
            lb_final=lb_starts[safe_d_idx],
            b1_final=b1_array[safe_d_idx],
            b2_final=b2_array[safe_d_idx],
            xs_final=xs_final,
            ws_pv_final=ws_pv_final,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    def multi_stage_from_mid(_):
        xs_mid_all, ws_mid_all, ws_pv_mid = first_stage_mid()
        ws_pv_mid = middle_stages(xs_mid_all, ws_mid_all, ws_pv_mid)
        return last_stage_from_mid(xs_mid_all, ws_pv_mid)

    if max_d == 2:
        return first_stage_last(None)
    return lax.cond(safe_d_idx == 1, first_stage_last, multi_stage_from_mid, operand=None)


def compute_heterog_multistage_logfptd_precomputed(
    rt,
    choice,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    d,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
):
    """Generalized multi-stage log-FPTD with per-stage sigma and boundary slopes.

    JAX uses a fixed-length series controlled only by ``trunc_num``. There is no
    ``adaptive_stopping`` or ``threshold`` option in this backend.

    This public wrapper keeps the generalized multistage signature while
    dispatching to the explicit precomputed production kernel.
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )

    def single_fn(_):
        return log_fptd_single(
            rt,
            mu_array[0],
            sigma_array[0],
            a1,
            b1_array[0],
            a2,
            b2_array[0],
            x0,
            choice,
            trunc_num=trunc_num,
        )

    if mu_array.shape[0] < 2:
        return single_fn(None)

    def multi_fn(_):
        return _heterog_multistage_logfptd_precomputed(
            rt,
            choice,
            x0,
            a1,
            a2,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    return lax.cond(d == 1, single_fn, multi_fn, operand=None)


# ---------------------------------------------------------------------------
# Single-trial stage-scan kernels
# ---------------------------------------------------------------------------


def _addm_logfptd_stagescan(
    rt,
    choice,
    sigma,
    a,
    b,
    x0,
    mu_array,
    sacc_array,
    d,
    *,
    order_mid,
    order_last,
    trunc_num,
    log_space,
):
    """Single-trial ADDM log-FPTD stage-scan kernel using scan-time transition updates.

    The computation skeleton is:

    1. build the per-stage boundary schedule
    2. place quadrature nodes and weights at the first interface
    3. advance stage by stage inside ``lax.scan``
    4. evaluate the final-stage boundary-hit density
    5. reduce over latent start positions in the final stage

    Notes
    -----
    This is the public single-trial stage-scan implementation for JAX
    multistage ADDM. Unlike :func:`_addm_logfptd_precomputed`, it does not
    materialize all stage transition matrices up front. Instead, each stage
    transition matrix is computed inside the ``lax.scan`` body and consumed
    immediately.

    That makes this path a useful correctness oracle:

    - same backend as the production JAX path
    - simpler stage-by-stage recurrence
    - lower conceptual coupling to precomputed ``P_all``

    It is usually slower than the production precomputed kernel and is kept as
    the explicit stage-scan alternative for tests, benchmarking, and clearer
    algorithmic comparison.
    """
    max_d = mu_array.shape[0]
    if max_d < 2:
        return log_fptd_single(
            rt, mu_array[0], sigma, a, -b, -a, b, x0, choice, trunc_num=trunc_num
        )
    x_ref_mid, w_ref_mid = get_gauss_legendre_ref(order_mid)
    x_ref_last, w_ref_last = get_gauss_legendre_ref(order_last)

    safe_stage_duration_array, upper_slope_array, lower_slope_array, a_starts = (
        _effective_addm_schedule(sacc_array, d, a, b)
    )
    safe_d_idx = jnp.minimum(d - 1, max_d - 1)

    def first_stage_last(_):
        xs_final, ws_final = _symmetric_stage_grid(x_ref_last, w_ref_last, a_starts[1])
        ws_pv_final = _addm_first_stage_to_grid(
            xs_final,
            ws_final,
            mu0=mu_array[0],
            sigma=sigma,
            a=a,
            upper_slope0=upper_slope_array[0],
            lower_slope0=lower_slope_array[0],
            stage_duration0=safe_stage_duration_array[0],
            x0=x0,
            trunc_num=trunc_num,
            log_space=log_space,
        )
        return _finish_addm_last_stage(
            rt - sacc_array[1],
            choice,
            mu_final=mu_array[1],
            sigma=sigma,
            a_final=a_starts[1],
            b=b,
            xs_final=xs_final,
            ws_pv_final=ws_pv_final,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    def first_stage_mid():
        a_init_mid = a_starts[1]
        xs_init_mid, ws_init_mid = _symmetric_stage_grid(
            x_ref_mid, w_ref_mid, a_init_mid
        )
        ws_pv_init_mid = _addm_first_stage_to_grid(
            xs_init_mid,
            ws_init_mid,
            mu0=mu_array[0],
            sigma=sigma,
            a=a,
            upper_slope0=upper_slope_array[0],
            lower_slope0=lower_slope_array[0],
            stage_duration0=safe_stage_duration_array[0],
            x0=x0,
            trunc_num=trunc_num,
            log_space=log_space,
        )
        return xs_init_mid, ws_pv_init_mid

    def middle_stages(carry):
        if log_space:

            def stage_step(carry, stage_idx):
                xs_prev_mid, log_ws_pv_prev_mid = carry
                a_prev_mid = a_starts[stage_idx]
                a_curr_mid = a_starts[stage_idx + 1]
                xs_mid = x_ref_mid * a_curr_mid
                ws_mid = w_ref_mid * a_curr_mid
                P_mid = q_single(
                    xs_mid[:, None],
                    mu_array[stage_idx],
                    sigma,
                    a_prev_mid,
                    upper_slope_array[stage_idx],
                    -a_prev_mid,
                    lower_slope_array[stage_idx],
                    safe_stage_duration_array[stage_idx],
                    xs_prev_mid[None, :],
                    trunc_num=trunc_num,
                )
                log_pv_mid = logsumexp(
                    positive_log(P_mid) + log_ws_pv_prev_mid[None, :], axis=1
                )
                log_ws_pv_mid = positive_log(ws_mid) + log_pv_mid
                active = stage_idx < (d - 2)
                xs_out = jnp.where(active, xs_mid, xs_prev_mid)
                ws_out = jnp.where(active, log_ws_pv_mid, log_ws_pv_prev_mid)
                return (xs_out, ws_out), None

        else:

            def stage_step(carry, stage_idx):
                xs_prev_mid, ws_pv_prev_mid = carry
                a_prev_mid = a_starts[stage_idx]
                a_curr_mid = a_starts[stage_idx + 1]
                xs_mid = x_ref_mid * a_curr_mid
                ws_mid = w_ref_mid * a_curr_mid
                P_mid = q_single(
                    xs_mid[:, None],
                    mu_array[stage_idx],
                    sigma,
                    a_prev_mid,
                    upper_slope_array[stage_idx],
                    -a_prev_mid,
                    lower_slope_array[stage_idx],
                    safe_stage_duration_array[stage_idx],
                    xs_prev_mid[None, :],
                    trunc_num=trunc_num,
                )
                pv_mid = P_mid @ ws_pv_prev_mid
                ws_pv_mid = ws_mid * pv_mid
                active = stage_idx < (d - 2)
                xs_out = jnp.where(active, xs_mid, xs_prev_mid)
                ws_out = jnp.where(active, ws_pv_mid, ws_pv_prev_mid)
                return (xs_out, ws_out), None

        if max_d > 3:
            carry, _ = lax.scan(stage_step, carry, jnp.arange(1, max_d - 2))
        return carry

    def last_stage_from_mid(xs_mid_final, ws_pv_mid_final):
        last_q_stage_idx = safe_d_idx - 1
        a_prev_last = a_starts[last_q_stage_idx]
        a_final = a_starts[safe_d_idx]
        xs_final = x_ref_last * a_final
        ws_final = w_ref_last * a_final
        P_last = q_single(
            xs_final[:, None],
            mu_array[last_q_stage_idx],
            sigma,
            a_prev_last,
            upper_slope_array[last_q_stage_idx],
            -a_prev_last,
            lower_slope_array[last_q_stage_idx],
            safe_stage_duration_array[last_q_stage_idx],
            xs_mid_final[None, :],
            trunc_num=trunc_num,
        )
        ws_pv_final = _apply_transition(P_last, ws_final, ws_pv_mid_final, log_space)
        return _finish_addm_last_stage(
            rt - sacc_array[safe_d_idx],
            choice,
            mu_final=mu_array[safe_d_idx],
            sigma=sigma,
            a_final=a_final,
            b=b,
            xs_final=xs_final,
            ws_pv_final=ws_pv_final,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    def multi_stage_from_mid(_):
        carry = first_stage_mid()
        xs_mid_final, ws_pv_mid_final = middle_stages(carry)
        return last_stage_from_mid(xs_mid_final, ws_pv_mid_final)

    return lax.cond(safe_d_idx == 1, first_stage_last, multi_stage_from_mid, operand=None)


def compute_addm_logfptd_stagescan(
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
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
):
    """Public single-trial ADDM log-FPTD stage-scan wrapper.

    This is the public stage-scan counterpart to
    :func:`compute_addm_logfptd_precomputed`. Like the rest of the public aDDM
    surface, it accepts the original aDDM parameters and covariates
    ``(eta, kappa, r1, r2, flag)`` rather than a derived ``mu_array``.

    The plain :func:`compute_addm_logfptd` alias continues to point to the
    precomputed production path.
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )
    mu_array = _build_addm_mu_array(
        eta,
        kappa,
        r1,
        r2,
        flag,
        d,
        sacc_array.shape[0],
        sacc_array.dtype,
    )

    def single_fn(_):
        return log_fptd_single(
            rt,
            mu_array[0],
            sigma,
            a,
            -b,
            -a,
            b,
            x0,
            choice,
            trunc_num=trunc_num,
        )

    if mu_array.shape[0] < 2:
        return single_fn(None)

    def multi_fn(_):
        return _addm_logfptd_stagescan(
            rt,
            choice,
            sigma,
            a,
            b,
            x0,
            mu_array,
            sacc_array,
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    return lax.cond(d == 1, single_fn, multi_fn, operand=None)


def _heterog_multistage_logfptd_stagescan(
    rt,
    choice,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    d,
    *,
    order_mid,
    order_last,
    trunc_num,
    log_space,
):
    """Single-trial generalized multistage log-FPTD stage-scan kernel.

    The computation skeleton mirrors :func:`_addm_logfptd_stagescan`, but tracks
    separate upper/lower boundary positions and per-stage sigma values.

    Notes
    -----
    This is the public stage-scan counterpart to
    :func:`_heterog_multistage_logfptd_precomputed`. It computes each stage
    transition matrix inside the scan body, which makes the execution order
    closer to the mathematical recurrence and easier to compare against other
    backends.
    """
    max_d = mu_array.shape[0]
    if max_d < 2:
        return log_fptd_single(
            rt,
            mu_array[0],
            sigma_array[0],
            a1,
            b1_array[0],
            a2,
            b2_array[0],
            x0,
            choice,
            trunc_num=trunc_num,
        )
    x_ref_mid, w_ref_mid = get_gauss_legendre_ref(order_mid)
    x_ref_last, w_ref_last = get_gauss_legendre_ref(order_last)

    (
        safe_stage_duration_array,
        upper_slope_array,
        lower_slope_array,
        ub_starts,
        lb_starts,
    ) = _effective_general_schedule(node_array, d, a1, b1_array, a2, b2_array)
    safe_d_idx = jnp.minimum(d - 1, max_d - 1)

    def first_stage_last(_):
        xs_final, ws_final = _general_stage_grid(
            x_ref_last, w_ref_last, ub_starts[1], lb_starts[1]
        )
        ws_pv_final = _general_first_stage_to_grid(
            xs_final,
            ws_final,
            x0=x0,
            mu0=mu_array[0],
            sigma0=sigma_array[0],
            upper_start0=a1,
            lower_start0=a2,
            upper_slope0=upper_slope_array[0],
            lower_slope0=lower_slope_array[0],
            stage_duration0=safe_stage_duration_array[0],
            trunc_num=trunc_num,
            log_space=log_space,
        )
        return _finish_general_last_stage(
            rt - node_array[1],
            choice,
            mu_final=mu_array[1],
            sigma_final=sigma_array[1],
            ub_final=ub_starts[1],
            lb_final=lb_starts[1],
            b1_final=b1_array[1],
            b2_final=b2_array[1],
            xs_final=xs_final,
            ws_pv_final=ws_pv_final,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    def first_stage_mid():
        ub_init_mid = ub_starts[1]
        lb_init_mid = lb_starts[1]
        xs_init_mid, ws_init_mid = _general_stage_grid(
            x_ref_mid, w_ref_mid, ub_init_mid, lb_init_mid
        )
        ws_pv_init_mid = _general_first_stage_to_grid(
            xs_init_mid,
            ws_init_mid,
            x0=x0,
            mu0=mu_array[0],
            sigma0=sigma_array[0],
            upper_start0=a1,
            lower_start0=a2,
            upper_slope0=upper_slope_array[0],
            lower_slope0=lower_slope_array[0],
            stage_duration0=safe_stage_duration_array[0],
            trunc_num=trunc_num,
            log_space=log_space,
        )
        return xs_init_mid, ws_pv_init_mid

    def middle_stages(carry):
        if log_space:

            def stage_step(carry, stage_idx):
                xs_prev_mid, log_ws_pv_prev_mid = carry
                ub_prev_mid = ub_starts[stage_idx]
                lb_prev_mid = lb_starts[stage_idx]
                ub_curr_mid = ub_starts[stage_idx + 1]
                lb_curr_mid = lb_starts[stage_idx + 1]
                half_w_mid = (ub_curr_mid - lb_curr_mid) / 2.0
                center_mid = (ub_curr_mid + lb_curr_mid) / 2.0
                xs_mid = x_ref_mid * half_w_mid + center_mid
                ws_mid = w_ref_mid * half_w_mid
                P_mid = q_single(
                    xs_mid[:, None],
                    mu_array[stage_idx],
                    sigma_array[stage_idx],
                    ub_prev_mid,
                    upper_slope_array[stage_idx],
                    lb_prev_mid,
                    lower_slope_array[stage_idx],
                    safe_stage_duration_array[stage_idx],
                    xs_prev_mid[None, :],
                    trunc_num=trunc_num,
                )
                log_pv_mid = logsumexp(
                    positive_log(P_mid) + log_ws_pv_prev_mid[None, :], axis=1
                )
                log_ws_pv_mid = positive_log(ws_mid) + log_pv_mid
                active = stage_idx < (d - 2)
                xs_out = jnp.where(active, xs_mid, xs_prev_mid)
                ws_out = jnp.where(active, log_ws_pv_mid, log_ws_pv_prev_mid)
                return (xs_out, ws_out), None

        else:

            def stage_step(carry, stage_idx):
                xs_prev_mid, ws_pv_prev_mid = carry
                ub_prev_mid = ub_starts[stage_idx]
                lb_prev_mid = lb_starts[stage_idx]
                ub_curr_mid = ub_starts[stage_idx + 1]
                lb_curr_mid = lb_starts[stage_idx + 1]
                half_w_mid = (ub_curr_mid - lb_curr_mid) / 2.0
                center_mid = (ub_curr_mid + lb_curr_mid) / 2.0
                xs_mid = x_ref_mid * half_w_mid + center_mid
                ws_mid = w_ref_mid * half_w_mid
                P_mid = q_single(
                    xs_mid[:, None],
                    mu_array[stage_idx],
                    sigma_array[stage_idx],
                    ub_prev_mid,
                    upper_slope_array[stage_idx],
                    lb_prev_mid,
                    lower_slope_array[stage_idx],
                    safe_stage_duration_array[stage_idx],
                    xs_prev_mid[None, :],
                    trunc_num=trunc_num,
                )
                pv_mid = P_mid @ ws_pv_prev_mid
                ws_pv_mid = ws_mid * pv_mid
                active = stage_idx < (d - 2)
                xs_out = jnp.where(active, xs_mid, xs_prev_mid)
                ws_out = jnp.where(active, ws_pv_mid, ws_pv_prev_mid)
                return (xs_out, ws_out), None

        if max_d > 3:
            carry, _ = lax.scan(stage_step, carry, jnp.arange(1, max_d - 2))
        return carry

    def last_stage_from_mid(xs_mid_final, ws_pv_mid_final):
        last_q_stage_idx = safe_d_idx - 1
        ub_prev_last = ub_starts[last_q_stage_idx]
        lb_prev_last = lb_starts[last_q_stage_idx]
        ub_final = ub_starts[safe_d_idx]
        lb_final = lb_starts[safe_d_idx]
        half_w_final = (ub_final - lb_final) / 2.0
        center_final = (ub_final + lb_final) / 2.0
        xs_final = x_ref_last * half_w_final + center_final
        ws_final = w_ref_last * half_w_final
        P_last = q_single(
            xs_final[:, None],
            mu_array[last_q_stage_idx],
            sigma_array[last_q_stage_idx],
            ub_prev_last,
            upper_slope_array[last_q_stage_idx],
            lb_prev_last,
            lower_slope_array[last_q_stage_idx],
            safe_stage_duration_array[last_q_stage_idx],
            xs_mid_final[None, :],
            trunc_num=trunc_num,
        )
        ws_pv_final = _apply_transition(P_last, ws_final, ws_pv_mid_final, log_space)
        return _finish_general_last_stage(
            rt - node_array[safe_d_idx],
            choice,
            mu_final=mu_array[safe_d_idx],
            sigma_final=sigma_array[safe_d_idx],
            ub_final=ub_starts[safe_d_idx],
            lb_final=lb_starts[safe_d_idx],
            b1_final=b1_array[safe_d_idx],
            b2_final=b2_array[safe_d_idx],
            xs_final=xs_final,
            ws_pv_final=ws_pv_final,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    def multi_stage_from_mid(_):
        carry = first_stage_mid()
        xs_mid_final, ws_pv_mid_final = middle_stages(carry)
        return last_stage_from_mid(xs_mid_final, ws_pv_mid_final)

    return lax.cond(safe_d_idx == 1, first_stage_last, multi_stage_from_mid, operand=None)


def compute_heterog_multistage_logfptd_stagescan(
    rt,
    choice,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    d,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    trunc_num=DEFAULT_TRUNC_NUM,
    log_space=False,
):
    """Public single-trial generalized multistage log-FPTD stage-scan wrapper.

    This is the public stage-scan counterpart to
    :func:`compute_heterog_multistage_logfptd_precomputed`. The default alias
    :func:`compute_heterog_multistage_logfptd` still points to the precomputed
    production path.
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )

    def single_fn(_):
        return log_fptd_single(
            rt,
            mu_array[0],
            sigma_array[0],
            a1,
            b1_array[0],
            a2,
            b2_array[0],
            x0,
            choice,
            trunc_num=trunc_num,
        )

    if mu_array.shape[0] < 2:
        return single_fn(None)

    def multi_fn(_):
        return _heterog_multistage_logfptd_stagescan(
            rt,
            choice,
            x0,
            a1,
            a2,
            mu_array,
            node_array,
            sigma_array,
            b1_array,
            b2_array,
            d,
            order_mid=order_mid,
            order_last=order_last,
            trunc_num=trunc_num,
            log_space=log_space,
        )

    return lax.cond(d == 1, single_fn, multi_fn, operand=None)


# ---------------------------------------------------------------------------
# Public aliases and JIT wrappers
# ---------------------------------------------------------------------------


compute_addm_logfptd_precomputed_jit = jit(
    compute_addm_logfptd_precomputed,
    static_argnames=("order_mid", "order_last", "order", "trunc_num", "log_space"),
)
compute_addm_logfptd_stagescan_jit = jit(
    compute_addm_logfptd_stagescan,
    static_argnames=("order_mid", "order_last", "order", "trunc_num", "log_space"),
)
compute_heterog_multistage_logfptd_precomputed_jit = jit(
    compute_heterog_multistage_logfptd_precomputed,
    static_argnames=("order_mid", "order_last", "order", "trunc_num", "log_space"),
)
compute_heterog_multistage_logfptd_stagescan_jit = jit(
    compute_heterog_multistage_logfptd_stagescan,
    static_argnames=("order_mid", "order_last", "order", "trunc_num", "log_space"),
)

# Plain names keep pointing to the production precomputed kernels.
compute_addm_logfptd = compute_addm_logfptd_precomputed
compute_addm_logfptd_jit = compute_addm_logfptd_precomputed_jit
compute_heterog_multistage_logfptd = compute_heterog_multistage_logfptd_precomputed
compute_heterog_multistage_logfptd_jit = compute_heterog_multistage_logfptd_precomputed_jit
