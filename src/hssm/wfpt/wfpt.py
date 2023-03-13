"""
This module provides functions for producing for Wiener First-Passage Time (WFPT)
distributions that support arbitrary log-likelihood functions and random number
generation ops.
"""

from __future__ import annotations

from os import PathLike
from typing import Callable, Dict, List, Tuple, Type

import bambi as bmb
import numpy as np
import onnx
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy.typing import ArrayLike
from pytensor.tensor.random.op import RandomVariable
from ssms.basic_simulators import simulator  # type: ignore

from .base import log_pdf_sv
from .lan import make_jax_logp_funcs_from_onnx, make_jax_logp_ops, make_pytensor_logp

LogLikeFunc = Callable[..., ArrayLike]
LogLikeGrad = Callable[..., ArrayLike]


def make_wfpt_rv(list_params: List[str]) -> Type[RandomVariable]:
    """Builds a RandomVariable Op according to the list of parameters.

    Args:
        list_params (List[str]): a list of str of all parameters for this RandomVariable

    Returns:
        Type[RandomVariable]: a class of RandomVariable that are to be used in
            a pm.Distribution.
    """

    # pylint: disable=W0511, R0903
    class WFPTRandomVariable(RandomVariable):
        """WFPT random variable"""

        name: str = "WFPT_RV"

        # NOTE: This is wrong at the moment, but necessary
        # to get around the support checking in PyMC that would result in error
        ndim_supp: int = 0

        ndims_params: List[int] = [0 for _ in list_params]
        dtype: str = "floatX"
        _print_name: Tuple[str, str] = ("WFPT", "\\operatorname{WFPT}")
        _list_params = list_params

        # pylint: disable=arguments-renamed,bad-option-value,W0221
        # NOTE: `rng`` now is a np.random.Generator instead of RandomState
        # since the latter is now deprecated from numpy
        @classmethod
        def rng_fn(  # type: ignore
            cls,
            rng: np.random.Generator,
            *args,
            model: str = "ddm",
            size: int = 500,
            theta: List[str] | None = None,
            **kwargs,
        ) -> np.ndarray:
            """Generates random variables from this distribution."""
            iinfo32 = np.iinfo(np.uint32)

            seed = rng.integers(0, iinfo32.max, dtype=np.uint32)

            # Uses a "theta" to specify how `theta` is passed to
            # the simulator object
            if theta is not None:
                dict_params = dict(zip(cls._list_params, args))
                theta = [dict_params[param] for param in theta]
            else:
                theta = list(args)

            # Because the `theta` parameter requires

            sim_out = simulator(
                theta=theta, model=model, n_samples=size, random_state=seed, **kwargs
            )
            output = np.column_stack([sim_out["rts"], sim_out["choices"]])
            return output

    return WFPTRandomVariable


def make_distribution(
    loglik: LogLikeFunc | pytensor.graph.Op,
    list_params: List[str],
    rv: Type[RandomVariable] | None = None,
) -> Type[pm.Distribution]:
    """Constructs a pymc.Distribution from a log-likelihood function and a
    RandomVariable op.

    NOTE: We will gradually switch to the numpy-notype style.

    Parameters
    ----------
    loglik
        A loglikelihood function. It can be any Callable in Python.
    list_params
        A list of parameters that the log-likelihood accepts. The order of the
        parameters in the list will determine the order in which the parameters
        are passed to the log-likelihood function.
    rv
        A RandomVariable Op (a class, not an instance). If None, a default will be
        used.

    Returns
    -------
        A pymc.Distribution that uses the log-likelihood function.
    """

    random_variable = make_wfpt_rv(list_params) if not rv else rv

    class WFPTDistribution(pm.Distribution):
        """Wiener first-passage time (WFPT) log-likelihood for LANs."""

        # This is just a placeholder because pm.Distribution requires an rv_op
        # Might be updated in the future once

        # NOTE: rv_op is an INSTANCE of RandomVariable
        rv_op = random_variable()
        params = list_params

        @classmethod
        def dist(cls, **kwargs):  # pylint: disable=arguments-renamed
            dist_params = [
                pt.as_tensor_variable(pm.floatX(kwargs[param])) for param in cls.params
            ]
            other_kwargs = {k: v for k, v in kwargs.items() if k not in cls.params}
            return super().dist(dist_params, **other_kwargs)

        def logp(data, *dist_params):  # pylint: disable=E0213

            return loglik(data, *dist_params)

    return WFPTDistribution


WFPT = make_distribution(log_pdf_sv, ["v", "sv", "a", "z", "t"])


def make_lan_distribution(
    list_params: List[str],
    model: str | PathLike | onnx.ModelProto,
    rv: Type[RandomVariable] | None = None,
    backend: str | None = "pytensor",
) -> Type[pm.Distribution]:
    """Produces a PyMC distribution that uses the provided base or ONNX model as
    its log-likelihood function.

    Args:
        model: The path of the ONNX model, or one already loaded in memory.
        backend: Whether to use "pytensor" or "jax" as the backend of the
            log-likelihood computation. If `jax`, the function will be wrapped in an
            pytensor Op.
        list_params: A list of the names of the parameters following the order of
            how they are fed to the LAN.
        rv: The RandomVariable Op used for posterior sampling.
    Returns:
        A PyMC Distribution class that uses the ONNX model as its log-likelihood
        function.
    """
    if isinstance(model, (str, PathLike)):
        model = onnx.load(str(model))
    if backend == "pytensor":
        lan_logp_aes = make_pytensor_logp(model)
        return make_distribution(lan_logp_aes, list_params, rv)

    if backend == "jax":
        logp, logp_grad, logp_nojit = make_jax_logp_funcs_from_onnx(
            model,
            n_params=len(list_params),
        )
        lan_logp_jax = make_jax_logp_ops(logp, logp_grad, logp_nojit)
        return make_distribution(lan_logp_jax, list_params, rv)

    raise ValueError("Currently only 'pytensor' and 'jax' backends are supported.")


def make_family(
    dist: Type[pm.Distribution],
    list_params: List[str],
    link: str | Dict[str, bmb.families.Link],
    parent: str = "v",
    likelihood_name: str = "WFPT Likelihood",
    family_name="WFPT Family",
) -> bmb.Family:
    """Builds a family in bambi.

    Args:
        dist (pm.Distribution): a pm.Distribution class (not an instance).
        list_params (List[str]): a list of parameters for the likelihood function.
        link (str | Dict[str, bmb.families.Link]): a link function or a dictionary of
            parameter: link functions.
        parent (str): the parent parameter of the likelihood function. Defaults to v.
        likelihood_name (str): the name of the likelihood function.
        family_name (str): the name of the family.

    Returns:
        bmb.Family: an instance of a bambi family.
    """

    likelihood = bmb.Likelihood(
        likelihood_name, parent=parent, params=list_params, dist=dist
    )

    family = bmb.Family(family_name, likelihood=likelihood, link=link)

    return family
