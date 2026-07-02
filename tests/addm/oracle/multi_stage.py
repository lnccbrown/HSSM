# efpt NumPy reference backend -- vendored TEST ORACLE (not shipped in the package).
# Source: efficient-fpt @ d97a451479141acef845195610f0f9d85824844e
#         MIT License, Copyright (c) 2025 Sicheng Liu.
# Vendored verbatim except import retargeting to flatten the backend into this
# self-contained  package. Kept at efpt's ORIGINAL DEFAULT_TRUNC_NUM=100
# so it is an INDEPENDENT numerical reference for the HSSM-vendored jax kernel
# (do not sync to that kernel's local modifications).

import numpy as np
from .single_stage import fptd_single, q_single
from ._numpy_utils import positive_log
from .validation import check_multistage_params
from ._defaults import (
    DEFAULT_LAST_QUAD_ORDER,
    DEFAULT_MID_QUAD_ORDER,
    DEFAULT_TRUNC_NUM,
    DEFAULT_THRESHOLD,
)
from .quadrature import lgwt_lookup_table
from ._shared_utils import resolve_quadrature_orders


def _logsumexp(a, axis=None):
    """NumPy-only logsumexp specialized for the repo's reduction patterns."""
    a = np.asarray(a, dtype=np.float64)
    if axis is None:
        if a.size == 0:
            return -np.inf
        max_val = np.max(a)
        if np.isneginf(max_val):
            return -np.inf
        with np.errstate(invalid="ignore"):
            return max_val + np.log(np.sum(np.exp(a - max_val)))

    max_val = np.max(a, axis=axis, keepdims=True)
    with np.errstate(invalid="ignore"):
        out = max_val + np.log(np.sum(np.exp(a - max_val), axis=axis, keepdims=True))
    out = np.where(np.isneginf(max_val), -np.inf, out)
    return np.squeeze(out, axis=axis)

def compute_homog_multistage_logfptds_and_lognpd(
    t_grid,
    T,
    x0,
    a1,
    a2,
    mu_array,
    node_array,
    sigma_array,
    b1_array,
    b2_array,
    *,
    order_mid=DEFAULT_MID_QUAD_ORDER,
    order_last=DEFAULT_LAST_QUAD_ORDER,
    order=None,
    eps=1e-3,
    trunc_num=DEFAULT_TRUNC_NUM,
    threshold=DEFAULT_THRESHOLD,
    adaptive_stopping=True,
    log_space=False,
):
    """
    Computes log-FPTDs for a multistage DDM on a specified time grid.

    Parameters
    ----------
    t_grid : array-like
        Time grid at which to evaluate the first-passage time density.

    mu_array : array-like of shape (d,)
        Drift rates for each of the `d` stages.

    node_array : array-like of shape (d,)
        Start times of each stage. Must satisfy ``node_array[0] == 0``.

    sigma_array : array-like of shape (d,)
        Diffusion coefficients for each stage.

    a1 : float
        Initial position of the upper boundary at time 0.

    b1_array : array-like of shape (d,)
        Slopes of the upper boundary in each stage (boundary evolves linearly).

    a2 : float
        Initial position of the lower boundary at time 0.

    b2_array : array-like of shape (d,)
        Slopes of the lower boundary in each stage (boundary evolves linearly).

    T : float
        Final time of the simulation. This defines the end of the last stage.

    x0 : callable or 2D np.ndarray
        Initial distribution of the diffusion process:
        - If callable, represents a sub-probability density function p(x_0) over the initial state.
        - If a 2D array of shape (2, N), X(0) is a mixture of N point masses.
          where the first row contains weights and the second row contains support points.

    order_mid : int, optional (default=20)
        Quadrature order used for intermediate-stage `q_single` propagation.

    order_last : int, optional (default=30)
        Quadrature order used for stage-local `fptd_single` reductions.

    order : int, optional
        Legacy compatibility alias. If provided on its own, it sets both
        `order_mid` and `order_last` to the same value.

    eps : float, optional (default=1e-3)
        Tolerance for ignoring grid points in `t_grid` that are too close to `node_array`,
        to avoid numerical instability.

    trunc_num : int, optional (default=100)
        Maximum number of terms in the truncated series expansion used for single-stage computations.

    threshold : float, optional (default=1e-20)
        Early-stopping tolerance for the single-stage series expansion.

    adaptive_stopping : bool, optional (default=True)
        If True, stop early when the current term is smaller than `threshold`.
        If False, always compute exactly `trunc_num` terms.

    log_space : bool, optional (default=False)
        If True, use log-space computation to prevent underflow in deep
        multi-stage models. Uses logsumexp for numerically stable accumulation.

    Returns
    -------
    logfptd : np.ndarray of shape (3, len(t_grid))
        A matrix where:
        - Row 0: Filtered time grid
        - Row 1: log-FPTD values at the upper boundary
        - Row 2: log-FPTD values at the lower boundary

    final_state : np.ndarray of shape (2, N)
        The final (post-last-stage) log distribution over the process state.
        - Row 0: Grid of support points for the process state
        - Row 1: Corresponding log subdensity mass values at those points
    """
    order_mid, order_last = resolve_quadrature_orders(
        order_mid=order_mid,
        order_last=order_last,
        order=order,
    )

    # Check parameters
    mu_array = mu_array[node_array < T]
    sigma_array = sigma_array[node_array < T]
    b1_array = b1_array[node_array < T]
    b2_array = b2_array[node_array < T]
    node_array = node_array[node_array < T]
    d = len(mu_array)  # Number of stages

    ##### ASSERTIONS #####
    check_multistage_params(
        mu_array, node_array, sigma_array, a1, b1_array, a2, b2_array
    )
    ##### END OF ASSERTIONS #####

    # Initialize
    ub, lb = a1, a2
    if isinstance(x0, np.ndarray) and x0.ndim == 2:
        xs_eval_prev = x0[1]
        ws_eval_prev = x0[0]
        qs_eval_prev = np.ones_like(ws_eval_prev)
        xs_prop_prev = xs_eval_prev
        ws_prop_prev = ws_eval_prev
        qs_prop_prev = qs_eval_prev
    elif callable(x0):
        xs_eval_prev, ws_eval_prev = lgwt_lookup_table(order_last, lb, ub)
        qs_eval_prev = x0(xs_eval_prev)
        xs_prop_prev, ws_prop_prev = lgwt_lookup_table(order_mid, lb, ub)
        qs_prop_prev = x0(xs_prop_prev)
    else:
        raise TypeError("x0 must be either a callable or a 2D point-mass array")

    ub_prev, lb_prev = ub, lb
    _node_array = np.concatenate([node_array, [T]])

    # skip t that are too close to `node_array` to avoid numerical instability issues
    t_grid, indices, _ = filter_and_group(_node_array, t_grid, epsilon=eps)
    upper_logfptds = np.full_like(t_grid, -np.inf)
    lower_logfptds = np.full_like(t_grid, -np.inf)

    if log_space:
        log_ws_qs_eval_prev = positive_log(ws_eval_prev * qs_eval_prev)
        log_ws_qs_prop_prev = positive_log(ws_prop_prev * qs_prop_prev)

    for n in range(d):
        ub += b1_array[n] * (_node_array[n + 1] - _node_array[n])
        lb += b2_array[n] * (_node_array[n + 1] - _node_array[n])
        xs_prop_src = xs_prop_prev
        ws_prop_src = ws_prop_prev
        qs_prop_src = qs_prop_prev
        if log_space:
            log_ws_qs_prop_src = log_ws_qs_prop_prev

        if len(indices[n]) > 0:
            U = fptd_single(
                t_grid[indices[n]][:, np.newaxis] - _node_array[n],
                mu_array[n],
                sigma_array[n],
                ub_prev,
                b1_array[n],
                lb_prev,
                b2_array[n],
                xs_eval_prev,
                1,
                trunc_num=trunc_num,
                threshold=threshold,
                adaptive_stopping=adaptive_stopping,
            )
            L = fptd_single(
                t_grid[indices[n]][:, np.newaxis] - _node_array[n],
                mu_array[n],
                sigma_array[n],
                ub_prev,
                b1_array[n],
                lb_prev,
                b2_array[n],
                xs_eval_prev,
                -1,
                trunc_num=trunc_num,
                threshold=threshold,
                adaptive_stopping=adaptive_stopping,
            )
            if log_space:
                # log-space accumulation for FPTD
                log_U = positive_log(U)
                log_L = positive_log(L)
                upper_logfptds[indices[n]] = _logsumexp(
                    log_U + log_ws_qs_eval_prev[np.newaxis, :], axis=1
                )
                lower_logfptds[indices[n]] = _logsumexp(
                    log_L + log_ws_qs_eval_prev[np.newaxis, :], axis=1
                )
            else:
                upper_logfptds[indices[n]] = positive_log(
                    np.sum(ws_eval_prev * qs_eval_prev * U, axis=1)
                )
                lower_logfptds[indices[n]] = positive_log(
                    np.sum(ws_eval_prev * qs_eval_prev * L, axis=1)
                )

        xs_prop, ws_prop = lgwt_lookup_table(order_mid, lb, ub)
        P_prop = q_single(
            xs_prop[:, np.newaxis],
            mu_array[n],
            sigma_array[n],
            ub_prev,
            b1_array[n],
            lb_prev,
            b2_array[n],
            _node_array[n + 1] - _node_array[n],
            xs_prop_src,
            trunc_num=trunc_num,
            threshold=threshold,
            adaptive_stopping=adaptive_stopping,
        )

        if log_space:
            log_P_prop = positive_log(P_prop)
            log_qs_prop = _logsumexp(
                log_P_prop + log_ws_qs_prop_src[np.newaxis, :], axis=1
            )
            log_ws_qs_prop_prev = positive_log(ws_prop) + log_qs_prop
            qs_prop_prev = log_qs_prop
        else:
            qs_prop_prev = np.sum(ws_prop_src * qs_prop_src * P_prop, axis=1)

        xs_prop_prev, ws_prop_prev = xs_prop, ws_prop

        if n < d - 1:
            xs_eval, ws_eval = lgwt_lookup_table(order_last, lb, ub)
            P_eval = q_single(
                xs_eval[:, np.newaxis],
                mu_array[n],
                sigma_array[n],
                ub_prev,
                b1_array[n],
                lb_prev,
                b2_array[n],
                _node_array[n + 1] - _node_array[n],
                xs_prop_src,
                trunc_num=trunc_num,
                threshold=threshold,
                adaptive_stopping=adaptive_stopping,
            )

            if log_space:
                log_P_eval = positive_log(P_eval)
                log_qs_eval = _logsumexp(
                    log_P_eval + log_ws_qs_prop_src[np.newaxis, :], axis=1
                )
                log_ws_qs_eval_prev = positive_log(ws_eval) + log_qs_eval
                qs_eval_prev = log_qs_eval
            else:
                qs_eval_prev = np.sum(ws_prop_src * qs_prop_src * P_eval, axis=1)

            xs_eval_prev, ws_eval_prev = xs_eval, ws_eval

        ub_prev, lb_prev = ub, lb

    log_qs_out = qs_prop_prev if log_space else positive_log(qs_prop_prev)

    return (
        np.vstack([t_grid, upper_logfptds, lower_logfptds]),
        np.vstack([xs_prop_prev, log_qs_out]),
    )


def filter_and_group(a, x, epsilon=1e-3):
    """
    Filters and groups values of `x` into open intervals defined by consecutive
    elements in `a`, excluding any values within `epsilon` of the interval boundaries.

    Parameters:
    ----------
    a : array-like of shape (d + 1,)
        Sorted array defining `d` open intervals of the form (a[i], a[i+1]).

    x : array-like
        Sorted array of values to be filtered and assigned to the intervals defined by `a`.

    epsilon : float, optional (default=1e-3)
        Tolerance for excluding values that are too close to the interval boundaries.

    Returns:
    -------
    filtered_x : np.ndarray
        Array of values from `x` that lie strictly inside one of the intervals
        (a[i], a[i+1]), excluding points near the boundaries.

    classified_indices : list of lists
        Each sublist contains indices of `filtered_x` that belong to interval i.

    classified_x : list of lists
        Each sublist contains the actual values from `x` that fall into interval i.
    """
    a = np.asarray(a, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    d = len(a) - 1

    # Filter: exclude x values within epsilon of any boundary in a
    dists = np.abs(x[:, np.newaxis] - a[np.newaxis, :])
    too_close = np.any(dists < epsilon, axis=1)
    keep = ~too_close

    # Assign each x to an interval via searchsorted (bins are [a[i], a[i+1]))
    bin_idx = np.searchsorted(a, x, side="right") - 1  # interval index
    in_range = (bin_idx >= 0) & (bin_idx < d)
    keep = keep & in_range

    filtered_x = x[keep]
    filtered_bins = bin_idx[keep]

    classified_indices = []
    classified_x = []
    for i in range(d):
        mask = filtered_bins == i
        classified_indices.append(np.where(mask)[0])
        classified_x.append(filtered_x[mask])

    return filtered_x, classified_indices, classified_x
