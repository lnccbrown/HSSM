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


def k_small(rt: np.ndarray, err: float) -> np.ndarray:
    """Determine number of terms needed for small-t expansion.

    Parameters
    ----------
    rt
        A 1D numpy array of flipped R.... T.....s. (0, inf).
    err
        Error bound.

    Returns
    -------
    np.ndarray
        A 1D at array of k_small.
    """
    ks = 2 + pt.sqrt(-2 * rt * pt.log(2 * np.sqrt(2 * np.pi * rt) * err))
    ks = pt.max(pt.stack([ks, pt.sqrt(rt) + 1]), axis=0)
    ks = pt.switch(2 * pt.sqrt(2 * np.pi * rt) * err < 1, ks, 2)

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
    kl = pt.sqrt(-2 * pt.log(np.pi * rt * err) / (np.pi**2 * rt))
    kl = pt.max(pt.stack([kl, 1.0 / (np.pi * pt.sqrt(rt))]), axis=0)
    kl = pt.switch(np.pi * rt * err < 1, kl, 1.0 / (np.pi * pt.sqrt(rt)))

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
    p = p / pt.sqrt(2 * np.pi * pt.power(tt, 3))

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
    y = k * pt.sin(k * np.pi * w)
    r = -pt.power(k, 2) * pt.power(np.pi, 2) * tt / 2
    p = pt.sum(y * pt.exp(r), axis=0) * np.pi

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
    data = pt.reshape(data, (-1, 2))
    rt = pt.abs(data[:, 0])
    response = data[:, 1]
    flip = response > 0
    a = a * 2.0
    v_flipped = pt.switch(flip, -v, v)  # transform v if x is upper-bound response
    z_flipped = pt.switch(flip, 1 - z, z)  # transform z if x is upper-bound response
    rt = rt - t

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

    data = pt.reshape(data, (-1, 2))
    rt = pt.abs(data[:, 0])
    response = data[:, 1]
    flip = response > 0
    a = a * 2.0
    v_flipped = pt.switch(flip, -v, v)  # transform v if x is upper-bound response
    z_flipped = pt.switch(flip, 1 - z, z)  # transform z if x is upper-bound response
    rt = rt - t

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
    "ddm",
    logp_ddm,
    list_params=["v", "a", "z", "t"],
    bounds=ddm_bounds,
)

DDM_SDV: Type[pm.Distribution] = make_distribution(
    "ddm_sdv",
    logp_ddm_sdv,
    list_params=ddm_sdv_params,
    bounds=ddm_sdv_bounds,
)
