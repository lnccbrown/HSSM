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


def make_rv(list_params: List[str]) -> Type[RandomVariable]:
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
        def rng_fn(  # type: ignore
            self,
            rng: np.random.Generator,
            *args,
            model: str = "ddm",
            size: int = 500,
            **kwargs,
        ) -> np.ndarray:
            """Generates random variables from this distribution."""
            iinfo32 = np.iinfo(np.int32)
            seed = rng.integers(iinfo32.min, iinfo32.max, dtype=np.int32)

            # Uses a "theta" in `kwargs` to specify how `theta` is passed to
            # the simulator object
            if "theta" in kwargs:
                dict_params = dict(zip(self._list_params, args))
                theta = [dict_params[param] for param in kwargs["theta"]]
            else:
                theta = list(args)

            # Because the `theta` parameter requires

            sim_out = simulator(
                theta=theta, model=model, n_samples=size, random_state=seed, **kwargs
            )
            data_tmp = np.column_stack([sim_out["rts"], sim_out["choices"]])
            return data_tmp

    return WFPTRandomVariable


def make_distribution(
    loglik: LogLikeFunc | pytensor.graph.Op | None,
    rv: Type[RandomVariable] | None,
    list_params: List[str],
) -> Type[pm.Distribution]:

    if rv is None:
        rv = make_rv(list_params)

    class WFPTDistribution(pm.Distribution):
        """Wiener first-passage time (WFPT) log-likelihood for LANs."""

        # This is just a placeholder because pm.Distribution requires an rv_op
        # Might be updated in the future once

        # NOTE: replace this default when we have a better random number generation
        # method. This is here as a place holder.
        rv_op = rv
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


def make_ssm_distribution(
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
    if model == "base":
        return make_distribution(log_pdf_sv, rv, list_params)

    if isinstance(model, (str, PathLike)):
        model = onnx.load(str(model))
    if backend == "pytensor":
        lan_logp_aes = make_pytensor_logp(model)
        return make_distribution(lan_logp_aes, rv, list_params)

    if backend == "jax":
        logp, logp_grad, logp_nojit = make_jax_logp_funcs_from_onnx(
            model,
            n_params=len(list_params),
        )
        lan_logp_jax = make_jax_logp_ops(logp, logp_grad, logp_nojit)
        return make_distribution(lan_logp_jax, rv, list_params)

    raise ValueError("Currently only 'pytensor' and 'jax' backends are supported.")


def make_family(
    dist: pm.Distribution,
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
