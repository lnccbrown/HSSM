"""Aesara functions for calculating the probability density of the Wiener
diffusion first-passage time (WFPT) distribution used in drift diffusion models (DDMs).

"""
from __future__ import annotations

from typing import List, Tuple

import aesara
import aesara.tensor as at

# import arviz as az
# import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from aesara.ifelse import ifelse
from aesara.tensor.random.op import RandomVariable
from pymc.distributions.continuous import PositiveContinuous

aesara.config.linker = "cvm"


def lambda_tt(tt: np.ndarray, err: float = 1e-7) -> Tuple[np.ndarray, np.ndarray]:
    """For each element in tt, return `True` if the large-time expansion is
    more efficient than the small-time expansion according to error bound
    err.

    Args:
        tt: A 1D numpy array of normalized RTs. (0, inf).
        err: Error bound.

    Return: A tuple of two arrays:
        lam: True` if the large-time expansion is more efficient than
            the small-time expansion according to error bound err.
        kappa: the kappa used to produce the number of terms in
            Navarro & Fuss (2009).
    """

    # determine number of terms needed for small-t expansion
    ks = 2 + np.sqrt(-2 * tt * np.log(2 * np.sqrt(2 * np.pi * tt) * err))
    ks = np.max(np.stack([ks, np.sqrt(tt) + 1]), axis=0)
    ks = np.where(2 * np.sqrt(2 * np.pi * tt) * err < 1, ks, 2)

    # determine number of terms needed for large-t expansion
    kl = np.sqrt(-2 * np.log(np.pi * tt * err) / (np.pi**2 * tt))
    kl = np.max(np.stack([kl, 1.0 / (np.pi * np.sqrt(tt))]), axis=0)
    kl = np.where(np.pi * tt * err < 1, kl, 1.0 / (np.pi * np.sqrt(tt)))

    lam = ks - kl >= 0

    # select the most accurate expansion for a fixed number of terms
    return lam, np.where(lam, kl, ks)


def n_terms(lam: np.ndarray, kappa: np.ndarray, terms: int | None = None) -> np.ndarray:
    """Computes a matrix of ks for vectorized summation of k terms
    determined by Eq. 13 in Navarro & Fuss (2009).

    Args:
        lam: Whether to use small or large time representation
        kappa: the kappa term in Eq. 13.
        term: Number of terms to include. If None, use the number of terms
            recommended in Navarro & Fuss (2009).

    Returns: a matrix of ks for vectorized summation of k terms
        determined by Eq. 13 in Navarro & Fuss (2009).
    """

    kappa_low = np.where(lam, 1, -np.floor((kappa - 1) / 2))

    if terms is None:

        kappa_high = np.where(lam, kappa, np.ceil((kappa - 1) / 2))
        ncols = np.max(kappa_high - kappa_low + 1)
        k_terms = np.repeat(np.arange(ncols).reshape((1, -1)), len(kappa), axis=0)

        k_terms = k_terms + kappa_low.reshape((-1, 1))
        k_mask = k_terms > kappa_high.reshape((-1, 1))
        k_terms = k_terms[k_mask]

        return k_terms, k_mask

    k_terms = np.repeat(np.arange(terms).reshape((1, -1)), len(kappa), axis=0)
    k_terms = k_terms + kappa_low.reshape((-1, 1))

    return k_terms, None


def ftt01w(tt: np.ndarray, w: np.ndarray, err: float = 1e-7) -> np.ndarray:
    """For each element in `x`, return `True` if the large-time expansion is
    more efficient than the small-time expansion.

    Args:
        tt: A 1D numpy array of normalized RTs. (0, inf).
        w: w = z / a is a relative start point.
        err: Error bound.

    Returns: the result of Eq. 13 in Navarro & Fuss, 2009
    """
    lam, kappa = lambda_tt(tt, err)
    k, k_mask = n_terms(lam, kappa)

    p_fast = (w + 2 * k) * at.exp((w + 2 * k) ** 2 / (2 * tt))
    p_fast = ifelse(k_mask is None, p_fast, k_mask * p_fast + (1 - k_mask) * p_fast)
    p_fast = at.sum(p_fast, axis=1) / at.sqrt(2 * np.pi * at.power(tt, 3))

    p_slow = k * at.exp(-(k**2) * np.pi**2 * tt / 2) * at.sin(k * np.pi * w)
    p_slow = ifelse(k_mask is None, p_fast, k_mask * p_slow + (1 - k_mask) * p_slow)
    p_slow = np.pi * np.sum(p_slow, axis=1)

    p = lam * p_slow + (1 - lam) * p_fast

    return p


def pdf(
    x: np.ndarray,
    v: float,
    a: float,
    z: float,
    err: float = 1e-7,
) -> np.ndarray:
    """Compute the likelihood of the drift diffusion model f(t|v,a,z) using the method
    and implementation of Navarro & Fuss, 2009.

    Args:
        x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        v: Mean drift rate. (-inf, inf).
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        err: Error bound.
    """

    # use normalized time
    tt = x / a**2

    p = ftt01w(tt, z, err)

    # convert to f(t|v,a,w)
    return p * at.exp(-v * a * z - v**2 * x / 2.0) / a**2


def pdf_sv(
    x: np.ndarray,
    v: float,
    sv: float,
    a: float,
    z: float,
    err: float = 1e-7,
) -> np.ndarray:
    """Compute the likelihood of the drift diffusion model f(t|v,a,z) using the method
    and implementation of Navarro & Fuss, 2009.

    Args:
        x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of v.
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        err: Error bound.
    """

    if sv == 0:
        return pdf(x, v, a, z, err)

    tt = x / (pow(a, 2))  # use normalized time

    p = ftt01w(tt, z, err)  # get f(t|0,1,w)

    p = (
        at.exp(
            at.log(p)
            + ((a * z * sv) ** 2 - 2 * a * v * z - (v**2) * x)
            / (2 * (sv**2) * x + 2)
        )
        / at.sqrt((sv**2) * x + 1)
        / (a**2)
    )

    return at.gt(x, 0) * p


def pdf_sv_sz(x, v, sv, a, z, sz, t):
    """Probability density function with normally distributed drift rate and uniformly
    distributed starting point.

    Args:
        x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of drift rate. [0, inf), 0 if fixed.
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        sz: Range of starting points [0, 1), 0 if fixed.
        t: Non-decision time. [0, inf)

    """
    # uses eq. 4.1.14 from Press et al., numerical recipes: the art of scientific
    # computing, 2007 for a more efficient 10-step approximation to the integral based
    # on simpson's rule
    n = 10
    h = sz / n
    z0 = z - sz / 2

    f = 17 * pdf_sv(x, v, sv, a, z0, t)
    f = f + 59 * pdf_sv(x, v, sv, a, z0 + h, t)
    f = f + 43 * pdf_sv(x, v, sv, a, z0 + 2 * h, t)
    f = f + 49 * pdf_sv(x, v, sv, a, z0 + 3 * h, t)

    f = f + 48 * pdf_sv(x, v, sv, a, z0 + 4 * h, t)
    f = f + 48 * pdf_sv(x, v, sv, a, z0 + 5 * h, t)
    f = f + 48 * pdf_sv(x, v, sv, a, z0 + 6 * h, t)

    f = f + 49 * pdf_sv(x, v, sv, a, z0 + h * (n - 3), t)
    f = f + 43 * pdf_sv(x, v, sv, a, z0 + h * (n - 2), t)
    f = f + 59 * pdf_sv(x, v, sv, a, z0 + h * (n - 1), t)
    f = f + 17 * pdf_sv(x, v, sv, a, z0 + h * n, t)

    return f / 48 / n


# def aesara_pdf_sv(x, v, sv, a, z, t):
#     """Probability density function with drift rates normally distributed over trials.

#     Args:
#         x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
#         v: Mean drift rate. (-inf, inf).
#         sv: Standard deviation of drift rate. [0, inf), 0 if fixed.
#         a: Value of decision upper bound. (0, inf).
#         z: Normalized decision starting point. (0, 1).
#         t: Non-decision time. [0, inf)

#     """
#     flip = x > 0
#     v = flip * -v + (1 - flip) * v  # transform v if x is upper-bound response
#     z = flip * (1 - z) + (1 - flip) * z  # transform z if x is upper-bound response
#     x = at.abs(x)  # abs_olute rts
#     x = x - t  # remove nondecision time

#     tt = x / at.power(a, 2)  # "normalize" RTs
#     p = aesara_fnorm(tt, z)  # normalized densities

#     return (
#         at.exp(
#             at.log(p)
#             + ((a * z * sv) ** 2 - 2 * a * v * z - (v**2) * x)
#             / (2 * (sv**2) * x + 2)
#         )
#         / at.sqrt((sv**2) * x + 1)
#         / (a**2)
#     )


def pdf_sv_sz_st(x, v, sv, a, z, sz, t, st):
    """ "Probability density function with normally distributed drift rate, uniformly
    distributed starting point, and uniformly distributed nondecision time.

    Args:
        x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of drift rate. [0, inf), 0 if fixed.
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        sz: Range of starting points [0, 1), 0 if fixed.
        t: Nondecision time. [0, inf)
        st: Range of Nondecision points [0, 1), 0 if fixed.

    """
    # uses eq. 4.1.14 from Press et al., Numerical recipes: the art of scientific
    # computing, 2007 for a more efficient n-step approximation to the integral based
    # on simpson's rule when n is even and > 4.
    n = 10
    h = st / n
    t0 = t - st / 2

    f = 17 * pdf_sv_sz(x, v, sv, a, z, sz, t0)
    f = f + 59 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + h)
    f = f + 43 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + 2 * h)
    f = f + 49 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + 3 * h)

    f = f + 48 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + 4 * h)
    f = f + 48 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + 5 * h)
    f = f + 48 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + 6 * h)

    f = f + 49 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + h * (n - 3))
    f = f + 43 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + h * (n - 2))
    f = f + 59 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + h * (n - 1))
    f = f + 17 * pdf_sv_sz(x, v, sv, a, z, sz, t0 + h * n)

    return f / 48 / n


def pdf_contaminant(x, l, r):
    """Probability density function of exponentially distributed RTs.

    Args:
        x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        l: Shape parameter of the exponential distribution. (0, inf).
        r: Proportion of upper-bound contaminants. [0, 1].

    """
    p = l * at.exp(-l * at.abs(x))
    return r * p * (x > 0) + (1 - r) * p * (x < 0)


def pdf_sv_sz_st_con(x, v, sv, a, z, sz, t, st, q, l, r):
    """ "Probability density function with normally distributed drift rate, uniformly
    distributed starting point, uniformly distributed nondecision time, and
    exponentially distributed contaminants.

    Args:
        x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of drift rate. [0, inf), 0 if fixed.
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        sz: Range of starting points [0, 1), 0 if fixed.
        t: Nondecision time. [0, inf)
        st: Range of Nondecision points [0, 1), 0 if fixed.
        q: Proportion of contaminants. [0, 1].
        l: Shape parameter of the exponential distribution. (0, inf).
        r: Proportion of upper-bound contaminants. [0, 1].

    """
    p_rts = pdf_sv_sz_st(x, v, sv, a, z, sz, t, st)
    p_con = pdf_contaminant(x, l, r)
    return p_rts * (1 - q) + p_con * q


def aesara_wfpt_log_like(x, v, sv, a, z, sz, t, st, q, l, r):
    """Returns the log likelihood of the WFPT distribution with normally distributed
    drift rate, uniformly distributed starting point, uniformly distributed nondecision
    time, and exponentially distributed contaminants.

    Args:
        x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of drift rate. [0, inf), 0 if fixed.
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        sz: Range of starting points [0, 1), 0 if fixed.
        t: Nondecision time. [0, inf)
        st: Range of Nondecision points [0, 1), 0 if fixed.
        q: Proportion of contaminants. [0, 1].
        l: Shape parameter of the exponential distribution. (0, inf).
        r: Proportion of upper-bound contaminants. [0, 1].

    """
    return at.sum(at.log(pdf_sv_sz_st_con(x, v, sv, a, z, sz, t, st, q, l, r)))


def wfpt_rvs(rng, v, sv, a, z, sz, t, st, q, l, r, size):
    """Generate random samples from the WFPT distribution with normally distributed
    drift rate, uniformly distributed starting point, uniformly distributed nondecision
    time, and exponentially distributed contaminants.

    Args:
        rng: np.random.RandomState
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of drift rate. [0, inf), 0 if fixed.
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        sz: Range of starting points [0, 1), 0 if fixed.
        t: Nondecision time. [0, inf)
        st: Range of Nondecision points [0, 1), 0 if fixed.
        q: Proportion of contaminants. [0, 1].
        l: Shape parameter of the exponential distribution. (0, inf).
        r: Proportion of upper-bound contaminants. [0, 1].
        size: Number of samples to generate.

    """

    x = np.linspace(-12, 12, 48000)
    dx = x[1] - x[0]

    numerical_pdf = pdf_sv_sz_st_con(x, v, sv + 1, a, z, sz, t, st, q, l, r).eval()
    cdf = np.cumsum(numerical_pdf) * dx
    cdf = cdf / cdf.max()
    u = rng.random_sample(
        size,
    )
    ix = np.searchsorted(cdf, u)

    return x[ix]


class WFPTRandomVariable(RandomVariable):
    """WFPT random variable"""

    name: str = "WFPT_RV"
    ndim_supp: int = 0
    ndims_params: List[int] = [0] * 9
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("WFPT", "WFPT")

    @classmethod
    def rng_fn(
        cls, rng: np.random.RandomState, v, sv, a, z, sz, t, st, q, l, r, size
    ) -> np.ndarray:
        return wfpt_rvs(rng, v, sv, a, z, sz, t, st, q, l, r, size)


class WFPT(PositiveContinuous):
    """Wiener first-passage time (WFPT) log-likelihood."""

    rv_op = WFPTRandomVariable()

    @classmethod
    def dist(cls, v, sv, a, z, sz, t, st, q, l, r, **kwargs):
        v = v = at.as_tensor_variable(pm.floatX(v))
        sv = sv = at.as_tensor_variable(pm.floatX(sv))
        a = a = at.as_tensor_variable(pm.floatX(a))
        z = z = at.as_tensor_variable(pm.floatX(z))
        sz = sz = at.as_tensor_variable(pm.floatX(sz))
        t = t = at.as_tensor_variable(pm.floatX(t))
        st = st = at.as_tensor_variable(pm.floatX(st))
        q = q = at.as_tensor_variable(pm.floatX(q))
        l = l = at.as_tensor_variable(pm.floatX(l))
        r = r = at.as_tensor_variable(pm.floatX(r))
        return super().dist([v, sv, a, z, sz, t, st, q, l, r], **kwargs)

    def logp(self, value, v, sv, a, z, sz, t, st, q, l, r):
        logp_expression = aesara_wfpt_log_like(
            value,
            v,
            sv,
            a,
            z,
            sz,
            t,
            st,
            q,
            l,
            r,
        )

        # bounded_logp_expression = at.switch(at.gt(value >= 0),
        # logp_expression, -np.inf)

        return logp_expression

    # def random(self, point=None, size=None):
    #     v, sv, a, z, sz, t, st, q, l, r, _ = draw_values(
    #         [
    #             self.v,
    #             self.sv,
    #             self.a,
    #             self.z,
    #             self.sz,
    #             self.t,
    #             self.st,
    #             self.q,
    #             self.l,
    #             self.r,
    #         ],
    #         point=point,
    #         size=size,
    #     )
    #     return generate_samples(aesara_wfpt_rvs,
    # v, sv, a, z, sz, t, st, q, l, r, size)


def test():

    v = 1
    sv = 0
    a = 0.8
    z = 0.5
    sz = 0.0
    t = 0.0
    st = 0.0
    q = 0.0
    l = 0.5
    r = 0

    size = 100

    rng = np.random.RandomState(10)

    x = wfpt_rvs(rng, v, sv, a, z, sz, t, st, q, l, r, size)
    print(x)

    with pm.Model():

        v = pm.Normal(name="v")
        WFPT(
            name="x", v=v, sv=sv, a=a, z=z, sz=sz, t=t, st=st, q=q, l=l, r=r, observed=x
        )
        results = pm.sample(1000, return_inferencedata=True)
        print(results)
        # az.plot_trace(results)
        # plt.show()


if __name__ == "__main__":
    test()
