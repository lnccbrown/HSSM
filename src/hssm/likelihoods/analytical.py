"""pytensor implementation of the Wiener First Passage Time Distribution.

This code is based on Sam Mathias's Pytensor/Theano implementation
of the WFPT distribution here:
https://gist.github.com/sammosummo/c1be633a74937efaca5215da776f194b.
"""

from typing import Type

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy import inf
from pymc.distributions.dist_math import check_parameters

from ..distribution_utils.dist import make_distribution

LOGP_LB = pm.floatX(-66.1)

π = np.pi
τ = 2 * π
sqrt_τ = pt.sqrt(τ)
log_π = pt.log(π)
log_τ = pt.log(τ)
log_4 = pt.log(4)


def _max(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return pt.max(pt.stack([a, b]), axis=0)


def k_small(rt: np.ndarray, err: float) -> np.ndarray:
    """Determine number of terms needed for small-t expansion.

    Parameters
    ----------
    rt
        A 1D numpy array of flipped RTs. (0, inf).
    err
        Error bound.

    Returns
    -------
    np.ndarray
        A 1D at array of k_small.
    """
    sqrt_rt = pt.sqrt(rt)
    log_rt = pt.log(rt)
    rt_log_2_sqrt_τ_rt_times_2 = rt * (log_4 + log_τ + log_rt)

    ks = 2 + pt.sqrt(-err * rt_log_2_sqrt_τ_rt_times_2)
    ks = _max(ks, sqrt_rt + 1)

    condition = 2 * sqrt_τ * sqrt_rt * err < 1
    ks = pt.switch(condition, ks, 2)

    return ks


def k_large(rt: np.ndarray, err: float) -> np.ndarray:
    """Determine number of terms needed for large-t expansion.

    Parameters
    ----------
    rt
        An 1D numpy array of flipped RTs. (0, inf).
    err
        Error bound

    Returns
    -------
    np.ndarray
        A 1D at array of k_large.
    """
    log_rt = pt.log(rt)
    sqrt_rt = pt.sqrt(rt)
    log_err = pt.log(err)

    π_rt_err = π * rt * err
    π_sqrt_rt = π * sqrt_rt

    kl = pt.sqrt(-2 * (log_π + log_err + log_rt)) / π_sqrt_rt
    kl = _max(kl, 1.0 / pt.sqrt(π_sqrt_rt))
    kl = pt.switch(π_rt_err < 1, kl, 1.0 / π_sqrt_rt)

    return kl


def compare_k(rt: np.ndarray, err: float) -> np.ndarray:
    """Compute and compare k_small with k_large.

    Parameters
    ----------
    rt
        An 1D numpy of flipped RTs. (0, inf).
    err
        Error bound.

    Returns
    -------
    np.ndarray
        A 1D boolean at array of which implementation should be used.
    """
    ks = k_small(rt, err)
    kl = k_large(rt, err)

    return ks < kl


def get_ks(k_terms: int, fast: bool) -> np.ndarray:
    """Return an array of ks.

    Returns an array of ks given the number of terms needed to approximate the sum of
    the infinite series.

    Parameters
    ----------
    k_terms
        number of terms needed
    fast
        whether the function is used in the fast of slow expansion.

    Returns
    -------
    np.ndarray
        An array of ks.
    """
    ks = (
        pt.arange(-pt.floor((k_terms - 1) / 2), pt.ceil((k_terms - 1) / 2) + 1)
        if fast
        else pt.arange(1, k_terms + 1).reshape((-1, 1))
    )

    return ks.astype(pytensor.config.floatX)


def ftt01w_fast(tt: np.ndarray, w: float, k_terms: int) -> np.ndarray:
    """Perform fast computation of ftt01w.

    Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the fast-RT expansion.

    Parameters
    ----------
    tt
        Flipped, normalized RTs. (0, inf).
    w
        Normalized decision starting point. (0, 1).
    k_terms
        number of terms to use to approximate the PDF.

    Returns
    -------
    np.ndarray
        The approximated function f(tt|0, 1, w).
    """
    # Slightly changed the original code to mimic the paper and
    # ensure correctness
    k = get_ks(k_terms, fast=True)

    # A log-sum-exp trick is used here
    y = w + 2 * k.reshape((-1, 1))
    r = -pt.power(y, 2) / (2 * tt)
    c = pt.max(r, axis=0)
    p = pt.exp(c) * pt.sum(y * pt.exp(r - c), axis=0)
    # Normalize p
    p = p / pt.sqrt(2 * π * pt.power(tt, 3))

    return p


def ftt01w_slow(tt: np.ndarray, w: float, k_terms: int) -> np.ndarray:
    """Perform slow computation of ftt01w.

    Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the slow-RT expansion.

    Parameters
    ----------
    tt
        Flipped, normalized RTs. (0, inf).
    w
        Normalized decision starting point. (0, 1).
    k_terms
        number of terms to use to approximate the PDF.

    Returns
    -------
    np.ndarray
        The approximated function f(tt|0, 1, w).
    """
    k = get_ks(k_terms, fast=False)
    y = k * pt.sin(k * π * w)
    r = -pt.power(k, 2) * pt.power(π, 2) * tt / 2
    p = pt.sum(y * pt.exp(r), axis=0) * π

    return p


def ftt01w(
    rt: np.ndarray,
    a: float,
    w: float,
    err: float = 1e-7,
    k_terms: int = 10,
) -> np.ndarray:
    """Compute the approximate density of f(tt|0,1,w).

    Parameters
    ----------
    rt
        Flipped Response Rates. (0, inf).
    a
        Value of decision upper bound. (0, inf).
    w
        Normalized decision starting point. (0, 1).
    err
        Error bound.
    k_terms
        number of terms to use to approximate the PDF.

    Returns
    -------
    np.ndarray
        The Approximated density of f(tt|0,1,w).
    """
    tt = rt / a**2.0

    lambda_rt = compare_k(tt, err)

    p_fast = ftt01w_fast(tt, w, k_terms)
    p_slow = ftt01w_slow(tt, w, k_terms)

    p = pt.switch(lambda_rt, p_fast, p_slow)

    return p


def logp_ddm(
    data: np.ndarray,
    v: float,
    a: float,
    z: float,
    t: float,
    err: float = 1e-15,
    k_terms: int = 20,
    epsilon: float = 1e-15,
) -> np.ndarray:
    """Compute analytical likelihood for the DDM model with `sv`.

    Computes the log-likelihood of the drift diffusion model f(t|v,a,z) using
    the method and implementation of Navarro & Fuss, 2009.

    Parameters
    ----------
    data
        data: 2-column numpy array of (response time, response)
    v
        Mean drift rate. (-inf, inf).
    a
        Value of decision upper bound. (0, inf).
    z
        Normalized decision starting point. (0, 1).
    t
        Non-decision time [0, inf).
    err
        Error bound.
    k_terms
        number of terms to use to approximate the PDF.
    epsilon
        A small positive number to prevent division by zero or
        taking the log of zero.

    Returns
    -------
    np.ndarray
        The analytical likelihoods for DDM.
    """
    data = pt.reshape(data, (-1, 2)).astype(pytensor.config.floatX)
    rt = pt.abs(data[:, 0])
    response = data[:, 1]
    flip = response > 0
    a = a * 2.0
    v_flipped = pt.switch(flip, -v, v)  # transform v if x is upper-bound response
    z_flipped = pt.switch(flip, 1 - z, z)  # transform z if x is upper-bound response
    rt = rt - t
    negative_rt = rt < 0
    rt = pt.switch(negative_rt, epsilon, rt)

    p = pt.maximum(ftt01w(rt, a, z_flipped, err, k_terms), pt.exp(LOGP_LB))

    logp = pt.where(
        rt <= epsilon,
        LOGP_LB,
        pt.log(p)
        - v_flipped * a * z_flipped
        - (v_flipped**2 * rt / 2.0)
        - 2.0 * pt.log(a),
    )

    checked_logp = check_parameters(logp, a >= 0, msg="a >= 0")
    checked_logp = check_parameters(checked_logp, z >= 0, msg="z >= 0")
    checked_logp = check_parameters(checked_logp, z <= 1, msg="z <= 1")
    return checked_logp


def logp_ddm_sdv(
    data: np.ndarray,
    v: float,
    a: float,
    z: float,
    t: float,
    sv: float,
    err: float = 1e-15,
    k_terms: int = 20,
    epsilon: float = 1e-15,
) -> np.ndarray:
    """Compute the log-likelihood of the drift diffusion model f(t|v,a,z).

    Using the method and implementation of Navarro & Fuss, 2009.

    Parameters
    ----------
    data
        2-column numpy array of (response time, response)
    v
        Mean drift rate. (-inf, inf).
    a
        Value of decision upper bound. (0, inf).
    z
        Normalized decision starting point. (0, 1).
    t
        Non-decision time [0, inf).
    sv
        Standard deviation of the drift rate [0, inf).
    err
        Error bound.
    k_terms
        number of terms to use to approximate the PDF.
    epsilon
        A small positive number to prevent division by zero or taking the log of zero.

    Returns
    -------
    np.ndarray
        The log likelihood of the drift diffusion model with the standard deviation
        of sv.
    """
    if sv == 0:
        return logp_ddm(data, v, a, z, t, err, k_terms, epsilon)

    data = pt.reshape(data, (-1, 2)).astype(pytensor.config.floatX)
    rt = pt.abs(data[:, 0])
    response = data[:, 1]
    flip = response > 0
    a = a * 2.0
    v_flipped = pt.switch(flip, -v, v)  # transform v if x is upper-bound response
    z_flipped = pt.switch(flip, 1 - z, z)  # transform z if x is upper-bound response
    rt = rt - t
    negative_rt = rt < 0
    rt = pt.switch(negative_rt, epsilon, rt)

    p = pt.maximum(ftt01w(rt, a, z_flipped, err, k_terms), pt.exp(LOGP_LB))

    logp = pt.switch(
        rt <= epsilon,
        LOGP_LB,
        pt.log(p)
        + (
            (a * z_flipped * sv) ** 2
            - 2 * a * v_flipped * z_flipped
            - (v_flipped**2) * rt
        )
        / (2 * (sv**2) * rt + 2)
        - 0.5 * pt.log(sv**2 * rt + 1)
        - 2 * pt.log(a),
    )

    checked_logp = check_parameters(logp, a >= 0, msg="a >= 0")
    checked_logp = check_parameters(checked_logp, z >= 0, msg="z >= 0")
    checked_logp = check_parameters(checked_logp, z <= 1, msg="z <= 1")
    checked_logp = check_parameters(checked_logp, sv > 0, msg="sv > 0")
    return checked_logp


ddm_bounds = {
    "v": (-inf, inf),
    "a": (0.0, inf),
    "z": (0.0, 1.0),
    "t": (0.0, inf),
}
ddm_sdv_bounds = ddm_bounds | {"sv": (0.0, inf)}

ddm_params = ["v", "a", "z", "t"]
ddm_sdv_params = ddm_params + ["sv"]

DDM: Type[pm.Distribution] = make_distribution(
    rv="ddm",
    loglik=logp_ddm,
    list_params=ddm_params,
    bounds=ddm_bounds,
)

DDM_SDV: Type[pm.Distribution] = make_distribution(
    rv="ddm_sdv",
    loglik=logp_ddm_sdv,
    list_params=ddm_sdv_params,
    bounds=ddm_sdv_bounds,
)


# LBA
def _pt_normpdf(t):
    return (1 / pt.sqrt(2 * pt.pi)) * pt.exp(-(t**2) / 2)


def _pt_normcdf(t):
    return (1 / 2) * (1 + pt.erf(t / pt.sqrt(2)))


def _pt_tpdf(t, A, b, v, s):
    g = (b - A - t * v) / (t * s)
    h = (b - t * v) / (t * s)
    f = (
        -v * _pt_normcdf(g)
        + s * _pt_normpdf(g)
        + v * _pt_normcdf(h)
        - s * _pt_normpdf(h)
    ) / A
    return f


def _pt_tcdf(t, A, b, v, s):
    e1 = ((b - A - t * v) / A) * _pt_normcdf((b - A - t * v) / (t * s))
    e2 = ((b - t * v) / A) * _pt_normcdf((b - t * v) / (t * s))
    e3 = ((t * s) / A) * _pt_normpdf((b - A - t * v) / (t * s))
    e4 = ((t * s) / A) * _pt_normpdf((b - t * v) / (t * s))
    F = 1 + e1 - e2 + e3 - e4
    return F


def _pt_lba3_ll(t, ch, A, b, v0, v1, v2):
    s = 0.1
    __min = pt.exp(LOGP_LB)
    __max = pt.exp(-LOGP_LB)
    k = len([0, 1, 2])
    like = pt.zeros((*t.shape, k))
    running_idx = pt.arange(t.shape[0])

    like_1 = (
        _pt_tpdf(t, A, b, v0, s)
        * (1 - _pt_tcdf(t, A, b, v1, s))
        * (1 - _pt_tcdf(t, A, b, v2, s))
    )
    like_2 = (
        (1 - _pt_tcdf(t, A, b, v0, s))
        * _pt_tpdf(t, A, b, v1, s)
        * (1 - _pt_tcdf(t, A, b, v2, s))
    )
    like_3 = (
        (1 - _pt_tcdf(t, A, b, v0, s))
        * (1 - _pt_tcdf(t, A, b, v1, s))
        * _pt_tpdf(t, A, b, v2, s)
    )

    like = pt.stack([like_1, like_2, like_3], axis=-1)

    # One should RETURN this because otherwise it will be pruned from graph
    # like_printed = pytensor.printing.Print('like')(like)

    prob_neg = _pt_normcdf(-v0 / s) * _pt_normcdf(-v1 / s) * _pt_normcdf(-v2 / s)
    return pt.log(pt.clip(like[running_idx, ch] / (1 - prob_neg), __min, __max))


def _pt_lba2_ll(t, ch, A, b, v0, v1):
    s = 0.1
    __min = pt.exp(LOGP_LB)
    __max = pt.exp(-LOGP_LB)
    k = len([0, 1])
    like = pt.zeros((*t.shape, k))
    running_idx = pt.arange(t.shape[0])

    like_1 = _pt_tpdf(t, A, b, v0, s) * (1 - _pt_tcdf(t, A, b, v1, s))
    like_2 = (1 - _pt_tcdf(t, A, b, v0, s)) * _pt_tpdf(t, A, b, v1, s)

    like = pt.stack([like_1, like_2], axis=-1)

    # One should RETURN this because otherwise it will be pruned from graph
    # like_printed = pytensor.printing.Print('like')(like)

    prob_neg = _pt_normcdf(-v0 / s) * _pt_normcdf(-v1 / s)
    return pt.log(pt.clip(like[running_idx, ch] / (1 - prob_neg), __min, __max))


def logp_lba2(
    data: np.ndarray,
    A: float,
    b: float,
    v0: float,
    v1: float,
) -> np.ndarray:
    """Compute the log-likelihood of the LBA model with 2 drift rates."""
    data = pt.reshape(data, (-1, 2)).astype(pytensor.config.floatX)
    rt = pt.abs(data[:, 0])
    response = data[:, 1]
    response_int = pt.cast(response, "int32")
    logp = _pt_lba2_ll(rt, response_int, A, b, v0, v1).squeeze()
    checked_logp = check_parameters(logp, b > A, msg="b > A")
    return checked_logp


def logp_lba3(
    data: np.ndarray,
    A: float,
    b: float,
    v0: float,
    v1: float,
    v2: float,
) -> np.ndarray:
    """Compute the log-likelihood of the LBA model with 3 drift rates."""
    data = pt.reshape(data, (-1, 2)).astype(pytensor.config.floatX)
    rt = pt.abs(data[:, 0])
    response = data[:, 1]
    response_int = pt.cast(response, "int32")
    logp = _pt_lba3_ll(rt, response_int, A, b, v0, v1, v2).squeeze()
    checked_logp = check_parameters(logp, b > A, msg="b > A")
    return checked_logp


lba2_params = ["A", "b", "v0", "v1"]
lba3_params = ["A", "b", "v0", "v1", "v2"]

lba2_bounds = {
    "A": (0.0, inf),
    "b": (0.2, inf),
    "v0": (0.0, inf),
    "v1": (0.0, inf),
}

lba3_bounds = {
    "A": (0.0, inf),
    "b": (0.2, inf),
    "v0": (0.0, inf),
    "v1": (0.0, inf),
    "v2": (0.0, inf),
}

LBA2: Type[pm.Distribution] = make_distribution(
    rv="lba2",
    loglik=logp_lba2,
    list_params=lba2_params,
    bounds=lba2_bounds,
)

LBA3: Type[pm.Distribution] = make_distribution(
    rv="lba3",
    loglik=logp_lba3,
    list_params=lba3_params,
    bounds=lba3_bounds,
)
