"""
This file includes the WFPT class factory that produces WFPT classes that support
arbitrary log-likelihood functions and random number generation functions, and
provides utility functions for handling LANs.
"""

from __future__ import annotations

from os import PathLike
from typing import Callable, List, Tuple, Type

import numpy as np
import onnx
import pymc as pm
import pytensor
import pytensor.tensor as pt
from numpy.typing import ArrayLike
from pytensor.tensor.random.op import RandomVariable
from ssms.basic_simulators import simulator  # type: ignore

from .base import log_pdf_sv
from .lan import LAN

LogLikeFunc = Callable[..., ArrayLike]
LogLikeGrad = Callable[..., ArrayLike]

# pylint: disable=W0511, R0903
# This is just a placeholder to get the code to run at the moment
class WFPTRandomVariable(RandomVariable):
    """WFPT random variable"""

    name: str = "WFPT_RV"
    ndim_supp: int = 0
    ndims_params: List[int] = [0] * 10
    dtype: str = "floatX"
    _print_name: Tuple[str, str] = ("WFPT", "\\operatorname{WFPT}")

    @classmethod
    # pylint: disable=arguments-renamed,bad-option-value,W0221
    def rng_fn(  # type: ignore
        cls,
        list_params: List[str] | None,
        model: str = "ddm",
        size: int = 500,
        **kwargs,
    ) -> np.ndarray:
        """Generates random variables from this distribution."""
        sim_out = simulator(theta=list_params, model=model, n_samples=size, **kwargs)
        data_tmp = sim_out["rts"] * sim_out["choices"]
        return data_tmp.flatten()


class WFPT:
    """
    This is a class factory for producing for Wiener First-Passage Time (WFPT)
    distributionsthat supports arbitrary log-likelihood functions and random number
    generation ops.
    """

    @classmethod
    def make_distribution(
        cls,
        loglik: LogLikeFunc | pytensor.graph.Op | None,
        rv: Type[RandomVariable] | None,
        list_params: List[str] | None,
    ) -> Type[pm.Distribution]:
        class WFPTDistribution(pm.Distribution):
            """Wiener first-passage time (WFPT) log-likelihood for LANs."""

            # This is just a placeholder because pm.Distribution requires an rv_op
            # Might be updated in the future once

            # NOTE: replace this default when we have a better random number generation
            # method. This is here as a place holder.
            rv_op = rv() if rv is not None else WFPTRandomVariable()
            params = list_params

            @classmethod
            def dist(cls, **kwargs):  # pylint: disable=arguments-renamed
                dist_params = [
                    pt.as_tensor_variable(pm.floatX(kwargs[param]))
                    for param in cls.params
                ]
                other_kwargs = {k: v for k, v in kwargs.items() if k not in cls.params}
                return super().dist(dist_params, **other_kwargs)

            def logp(data, *dist_params):  # pylint: disable=E0213

                return loglik(data, *dist_params)

        return WFPTDistribution

    @classmethod
    def make_ssm_distribution(
        cls,
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
            compile_funcs: Whether the JAX functions should be compiled, if the backend
                is set to JAX.
        Returns:
            A PyMC Distribution class that uses the ONNX model as its log-likelihood
            function.
        """
        if model == "base":
            return cls.make_distribution(log_pdf_sv, rv, list_params)

        if isinstance(model, (str, PathLike)):
            model = onnx.load(str(model))
        if backend == "pytensor":
            lan_logp_aes = LAN.make_pytensor_logp(model)
            return cls.make_distribution(lan_logp_aes, rv, list_params)

        if backend == "jax":
            logp, logp_grad, logp_nojit = LAN.make_jax_logp_funcs_from_onnx(
                model,
                n_params=len(list_params),
            )
            lan_logp_jax = LAN.make_jax_logp_ops(logp, logp_grad, logp_nojit)
            return cls.make_distribution(lan_logp_jax, rv, list_params)

        raise ValueError("Currently only 'pytensor' and 'jax' backends are supported.")
