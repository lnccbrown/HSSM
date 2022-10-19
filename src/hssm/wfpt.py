"""
wfpt.py: aesara implementation of the Wiener First Passage Time Distribution

This code is based on Sam Mathias's Aesara/Theano implementation
of the WFPT distribution here:
https://gist.github.com/sammosummo/c1be633a74937efaca5215da776f194b

"""

from __future__ import annotations

from typing import Callable, List, Tuple

import aesara.tensor as at
import numpy as np
import pandas as pd
import pymc as pm
from aesara.tensor.random.op import RandomVariable
from pymc.distributions.continuous import PositiveContinuous


def decision_func() -> Callable[[np.ndarray, float], at.tensor]:
    """Produces a decision function that determines whether the pdf should be calculated
    with large-time or small-time expansion.

    Returns: A decision function with saved state to avoid repeated computation.
    """

    internal_rt = None
    internal_err = None
    internal_result = None

    def inner_func(rt: np.ndarray, err: float = 1e-7) -> at.tensor:
        """For each element in `rt`, return `True` if the large-time expansion is
        more efficient than the small-time expansion and `False` otherwise.

        Args:
            rt: An 1D numpy of flipped RTs. (0, inf).
            err: Error bound

        Returns: a 1D boolean at array of which implementation should be used.
        """

        nonlocal internal_rt
        nonlocal internal_err
        nonlocal internal_result

        if np.all(rt == internal_rt) and err == internal_err:
            return internal_result

        internal_rt = rt
        internal_err = err

        # determine number of terms needed for small-t expansion
        ks = 2 + at.sqrt(-2 * rt * at.log(2 * np.sqrt(2 * np.pi * rt) * err))
        ks = at.max(at.stack([ks, at.sqrt(rt) + 1]), axis=0)
        ks = at.switch(2 * at.sqrt(2 * np.pi * rt) * err < 1, ks, 2)

        # determine number of terms needed for large-t expansion
        kl = at.sqrt(-2 * at.log(np.pi * rt * err) / (np.pi**2 * rt))
        kl = at.max(at.stack([kl, 1.0 / (np.pi * at.sqrt(rt))]), axis=0)
        kl = at.switch(np.pi * rt * err < 1, kl, 1.0 / (np.pi * at.sqrt(rt)))

        lambda_rt = ks < kl

        internal_result = lambda_rt

        return lambda_rt

    return inner_func


# This decision function keeps an internal state of `tt`
# and does not repeat computation if a new `tt` passed to
# it is the same
decision = decision_func()


def ftt01w_fast(tt: at.tensor, w: float, k_terms: int) -> at.tensor:
    """Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the fast-RT expansion.

    Args:
        tt: Flipped, normalized RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).
        k_terms: number of terms to use to approximate the PDF.

    Returns:
        The approximated function f(tt|0, 1, w).
    """

    # Slightly changed the original code to mimic the paper and
    # ensure correctness
    k = at.arange(-at.floor((k_terms - 1) / 2), at.ceil((k_terms - 1) / 2) + 1)
    y = w + 2 * k.reshape((-1, 1))
    r = -at.power(y, 2) / 2 / tt
    c = at.max(r, axis=0)
    p = at.exp(c + at.log(at.sum(y * at.exp(r - c), axis=0)))
    p = p / at.sqrt(2 * np.pi * at.power(tt, 3))

    return p


def ftt01w_slow(tt: at.tensor, w: float, k_terms: int) -> at.tensor:
    """Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the slow-RT expansion.

    Args:
        tt: Flipped, normalized RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).
        k_terms: number of terms to use to approximate the PDF.

    Returns:
        The approximated function f(tt|0, 1, w).
    """

    k = at.arange(1, k_terms + 1).reshape((-1, 1))
    y = k * at.sin(k * np.pi * w)
    r = -at.power(k, 2) * at.power(np.pi, 2) * tt / 2
    p = at.sum(y * at.exp(r), axis=0) * np.pi

    return p


def ftt01w(
    rt: np.ndarray,
    a: float,
    w: float,
    err: float = 1e-7,
    k_terms: int = 10,
) -> at.tensor:
    """Compute the appproximated density of f(tt|0,1,w) using the method
    and implementation of Navarro & Fuss, 2009.

    Args:
        rt: Flipped RTs. (0, inf).
        a: Value of decision upper bound. (0, inf).
        w: Normalized decision starting point. (0, 1).
        err: Error bound.
        k_terms: number of terms to use to approximate the PDF.
    """
    lambda_rt = decision(rt, err)
    tt = rt / a**2

    p_fast = ftt01w_fast(tt, w, k_terms)
    p_slow = ftt01w_slow(tt, w, k_terms)

    return at.switch(lambda_rt, p_fast, p_slow)


def log_pdf_sv(
    data: np.ndarray,
    v: float,
    sv: float,
    a: float,
    z: float,
    t: float,
    err: float = 1e-7,
    k_terms: int = 10,
) -> at.scalar:
    """Computes the log-likelihood of the drift diffusion model f(t|v,a,z) using
    the method and implementation of Navarro & Fuss, 2009.

    Args:
        data: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of the drift rate [0, inf).
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        t: Non-decision time [0, inf).
        err: Error bound.
        k_terms: number of terms to use to approximate the PDF.
    """

    # First, flip data to positive
    flip = data > 0
    v = flip * -v + (1 - flip) * v  # transform v if x is upper-bound response
    z = flip * (1 - z) + (1 - flip) * z  # transform z if x is upper-bound response
    rt = np.abs(data)  # absolute rts
    rt = rt - t  # remove nondecision time

    p = ftt01w(rt, a, z, err, k_terms)

    p = at.sum(
        at.log(p)
        + ((a * z * sv) ** 2 - 2 * a * v * z - (v**2) * rt) / (2 * (sv**2) * rt + 2)
        - at.log(sv**2 * rt + 1) / 2
        - 2 * at.log(a)
    )

    return p


# TODO: Implement this class.
# This is just a placeholder to get the code to run at the moment
class WFPTRandomVariable(RandomVariable):
    """WFPT random variable"""

    name: str = "WFPT_RV"
    ndim_supp: int = 0
    ndims_params: List[int] = [0] * 10
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("WFPT", "WFPT")

    @classmethod
    def rng_fn(
        cls,  # rng: np.random.RandomState, v, sv, a, z, sz, t, st, q, l, r, size
    ) -> np.ndarray:
        """Generates WFPT random variables."""

        return NotImplementedError("Not Implemented")


class WFPT(PositiveContinuous):
    """Wiener first-passage time (WFPT) distribution"""

    rv_op = WFPTRandomVariable()

    @classmethod
    def dist(cls, v, sv, a, z, t, **kwargs):
        v = at.as_tensor_variable(pm.floatX(v))
        sv = at.as_tensor_variable(pm.floatX(sv))
        a = at.as_tensor_variable(pm.floatX(a))
        z = at.as_tensor_variable(pm.floatX(z))
        t = at.as_tensor_variable(pm.floatX(t))
        return super().dist([v, sv, a, z, t], **kwargs)

    def logp(data, v, sv, a, z, t, err=1e-7, k_terms=10):

        return log_pdf_sv(data, v, sv, a, z, t, err, k_terms)


def main():
    """Simple tests of the code here"""

    cavanaugh_data = pd.read_csv("../scratch/cavanagh_theta_nn.txt")

    with pm.Model():

        sv = 0
        a = 0.8
        z = 0.5
        t = 0.0

        v = pm.Normal(name="v")
        WFPT(name="x", v=v, sv=sv, a=a, z=z, t=t, observed=cavanaugh_data.rt.values)
        results = pm.sample(1000, return_inferencedata=True)
        print(results)


if __name__ == "__main__":
    main()
