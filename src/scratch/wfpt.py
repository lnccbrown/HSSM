"""Aesara/Theano functions for calculating the probability density of the Wiener
diffusion first-passage time (WFPT) distribution used in drift diffusion models (DDMs).

"""
from typing import List, Tuple

import aesara.tensor as at

# import arviz as az
# import matplotlib.pyplot as plt
import numpy as np
import pymc as pm
from aesara.tensor.random.op import RandomVariable
from numba import njit
from pymc.distributions.continuous import PositiveContinuous

# from in_jax import jax_wfpt_rvs


@njit
def use_fast(tt: np.ndarray, err: float = 1e-7) -> np.ndarray:
    """For each element in `x`, return `True` if the large-time expansion is
    more efficient than the small-time expansion.

    Args:
        tt: An 1D numpy array of normalized RTs. (0, inf).
        err: Error bound

    Returns: TBD

    """

    # determine number of terms needed for small-t expansion
    ks = 2 + np.sqrt(-2 * tt * np.log(2 * np.sqrt(2 * np.pi * tt) * err))
    ks = np.max(np.stack([ks, np.sqrt(tt) + 1]), axis=0)
    ks = np.where(2 * np.sqrt(2 * np.pi * tt) * err < 1, ks, 2)

    # determine number of terms needed for large-t expansion
    kl = np.sqrt(-2 * np.log(np.pi * tt * err) / (np.pi**2 * tt))
    kl = np.max(np.stack([kl, 1.0 / (np.pi * np.sqrt(tt))]), axis=0)
    kl = np.where(np.pi * tt * err < 1, kl, 1.0 / (np.pi * np.sqrt(tt)))

    lambda_tt = (
        ks - kl >= 0
    )  # select the most accurate expansion for a fixed number of terms

    kappa = np.where(lambda_tt, kl, ks)

    kappa_low = np.where(lambda_tt, 1, -np.floor((kappa - 1) / 2))
    kappa_high = np.where(lambda_tt, kappa, np.ceil((kappa - 1) / 2))

    ncols = np.max(kappa_high - kappa_low + 1)
    out = np.zeros((len(kappa), ncols), dtype=np.int32)

    for i, (low, high) in enumerate(zip(kappa_low, kappa_high)):
        out[i, high - low] = np.arange(low, high + 1, dtype=np.int32)

    return out


def ftt01w_fast(x, w):
    """Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the fast-RT expansion.

    Args:
        x: RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).

    """
    bigk = 7  # number of terms to evaluate; TODO: this needs tuning eventually

    # calculated the dumb way
    # p1 = (w + 2 * -3) * T.exp(-T.power(w + 2 * -3, 2) / 2 / x)
    # p2 = (w + 2 * -2) * T.exp(-T.power(w + 2 * -2, 2) / 2 / x)
    # p3 = (w + 2 * -1) * T.exp(-T.power(w + 2 * -1, 2) / 2 / x)
    # p4 = (w + 2 * 0) * T.exp(-T.power(w + 2 * 0, 2) / 2 / x)
    # p5 = (w + 2 * 1) * T.exp(-T.power(w + 2 * 1, 2) / 2 / x)
    # p6 = (w + 2 * 2) * T.exp(-T.power(w + 2 * 2, 2) / 2 / x)
    # p7 = (w + 2 * 3) * T.exp(-T.power(w + 2 * 3, 2) / 2 / x)
    # p = (p1 + p2 + p3 + p4 + p5 + p6 + p7) / T.sqrt(2 * np.pi * T.power(x, 3))

    # calculated the better way
    # k = (T.arange(bigk) - T.floor(bigk / 2)).reshape((-1, 1))
    # y = w + 2 * k
    # r = -T.power(y, 2) / 2 / x
    # p = T.sum(y * T.exp(r), axis=0) / T.sqrt(2 * np.pi * T.power(x, 3))

    # calculated using the "log-sum-exp trick" to reduce under/overflows
    k = at.arange(bigk) - at.floor(bigk / 2)
    y = w + 2 * k.reshape((-1, 1))
    r = -at.power(y, 2) / 2 / x
    c = at.max(r, axis=0)
    p = at.exp(c + at.log(at.sum(y * at.exp(r - c), axis=0)))
    p = p / at.sqrt(2 * np.pi * at.power(x, 3))

    return p


def ftt01w_slow(x, w):
    """Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, calculated using the slow-RT expansion.

    Args:
        x: RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).

    """
    bigk = 7  # number of terms to evaluate; TODO: this needs tuning eventually

    # calculated the dumb way
    # b = T.power(np.pi, 2) * x / 2
    # p1 = T.exp(-T.power(1, 2) * b) * T.sin(np.pi * w)
    # p2 = 2 * T.exp(-T.power(2, 2) * b) * T.sin(2 * np.pi * w)
    # p3 = 3 * T.exp(-T.power(3, 2) * b) * T.sin(3 * np.pi * w)
    # p4 = 4 * T.exp(-T.power(4, 2) * b) * T.sin(4 * np.pi * w)
    # p5 = 5 * T.exp(-T.power(5, 2) * b) * T.sin(5 * np.pi * w)
    # p6 = 6 * T.exp(-T.power(6, 2) * b) * T.sin(6 * np.pi * w)
    # p7 = 7 * T.exp(-T.power(7, 2) * b) * T.sin(7 * np.pi * w)
    # p = (p1 + p2 + p3 + p4 + p5 + p6 + p7) * np.pi
    # print(p)

    # calculated the better way
    k = at.arange(1, bigk + 1).reshape((-1, 1))
    y = k * at.sin(k * np.pi * w)
    r = -at.power(k, 2) * at.power(np.pi, 2) * x / 2
    p = at.sum(y * at.exp(r), axis=0) * np.pi
    # print(p)

    # calculated using the "log-sum-exp trick" to reduce under/overflows
    # k = T.arange(1, bigk + 1).reshape((-1, 1))
    # y = k * T.sin(k * np.pi * w)
    # r = -T.power(k, 2) * T.power(np.pi, 2) * x / 2
    # c = T.max(r, axis=0)
    # p = T.exp(c + T.log(T.sum(y * T.exp(r - c), axis=0))) * np.pi

    return p


def aesara_fnorm(x, w):
    """Density function for lower-bound first-passage times with drift rate set to 0 and
    upper bound set to 1, selecting the most efficient expansion per element in `x`.

    Args:
        x: RTs. (0, inf).
        w: Normalized decision starting point. (0, 1).

    """
    y = at.abs(x)
    f = ftt01w_fast(y, w)
    s = ftt01w_slow(y, w)
    u = use_fast(y)
    positive = x > 0
    return (f * u + s * (1 - u)) * positive


def aesara_pdf_sv(x, v, sv, a, z, t):
    """Probability density function with drift rates normally distributed over trials.

    Args:
        x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        v: Mean drift rate. (-inf, inf).
        sv: Standard deviation of drift rate. [0, inf), 0 if fixed.
        a: Value of decision upper bound. (0, inf).
        z: Normalized decision starting point. (0, 1).
        t: Non-decision time. [0, inf)

    """
    flip = x > 0
    v = flip * -v + (1 - flip) * v  # transform v if x is upper-bound response
    z = flip * (1 - z) + (1 - flip) * z  # transform z if x is upper-bound response
    x = at.abs(x)  # abs_olute rts
    x = x - t  # remove nondecision time

    tt = x / at.power(a, 2)  # "normalize" RTs
    p = aesara_fnorm(tt, z)  # normalized densities

    return (
        at.exp(
            at.log(p)
            + ((a * z * sv) ** 2 - 2 * a * v * z - (v**2) * x)
            / (2 * (sv**2) * x + 2)
        )
        / at.sqrt((sv**2) * x + 1)
        / (a**2)
    )


def aesara_pdf_sv_sz(x, v, sv, a, z, sz, t):
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

    f = 17 * aesara_pdf_sv(x, v, sv, a, z0, t)
    f = f + 59 * aesara_pdf_sv(x, v, sv, a, z0 + h, t)
    f = f + 43 * aesara_pdf_sv(x, v, sv, a, z0 + 2 * h, t)
    f = f + 49 * aesara_pdf_sv(x, v, sv, a, z0 + 3 * h, t)

    f = f + 48 * aesara_pdf_sv(x, v, sv, a, z0 + 4 * h, t)
    f = f + 48 * aesara_pdf_sv(x, v, sv, a, z0 + 5 * h, t)
    f = f + 48 * aesara_pdf_sv(x, v, sv, a, z0 + 6 * h, t)

    f = f + 49 * aesara_pdf_sv(x, v, sv, a, z0 + h * (n - 3), t)
    f = f + 43 * aesara_pdf_sv(x, v, sv, a, z0 + h * (n - 2), t)
    f = f + 59 * aesara_pdf_sv(x, v, sv, a, z0 + h * (n - 1), t)
    f = f + 17 * aesara_pdf_sv(x, v, sv, a, z0 + h * n, t)

    return f / 48 / n


def aesara_pdf_sv_sz_st(x, v, sv, a, z, sz, t, st):
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

    f = 17 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0)
    f = f + 59 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + h)
    f = f + 43 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + 2 * h)
    f = f + 49 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + 3 * h)

    f = f + 48 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + 4 * h)
    f = f + 48 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + 5 * h)
    f = f + 48 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + 6 * h)

    f = f + 49 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + h * (n - 3))
    f = f + 43 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + h * (n - 2))
    f = f + 59 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + h * (n - 1))
    f = f + 17 * aesara_pdf_sv_sz(x, v, sv, a, z, sz, t0 + h * n)

    return f / 48 / n


def aesara_pdf_contaminant(x, l, r):
    """Probability density function of exponentially distributed RTs.

    Args:
        x: RTs. (-inf, inf) except 0. Negative values correspond to the lower bound.
        l: Shape parameter of the exponential distribution. (0, inf).
        r: Proportion of upper-bound contaminants. [0, 1].

    """
    p = l * at.exp(-l * at.abs(x))
    return r * p * (x > 0) + (1 - r) * p * (x < 0)


def aesara_pdf_sv_sz_st_con(x, v, sv, a, z, sz, t, st, q, l, r):
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
    p_rts = aesara_pdf_sv_sz_st(x, v, sv, a, z, sz, t, st)
    p_con = aesara_pdf_contaminant(x, l, r)
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
    return at.sum(at.log(aesara_pdf_sv_sz_st_con(x, v, sv, a, z, sz, t, st, q, l, r)))


def aesara_wfpt_rvs(rng, v, sv, a, z, sz, t, st, q, l, r, size):
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

    pdf = aesara_pdf_sv_sz_st_con(x, v, sv + 1, a, z, sz, t, st, q, l, r)
    cdf = at.extra_ops.cumsum(pdf) * dx
    cdf = cdf / cdf.max()
    u = rng.random_sample(
        size,
    )
    ix = at.extra_ops.searchsorted(cdf, u)

    return x[ix.eval()]


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
        return aesara_wfpt_rvs(rng, v, sv, a, z, sz, t, st, q, l, r, size)


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

    x = aesara_wfpt_rvs(rng, v, sv, a, z, sz, t, st, q, l, r, size)
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
