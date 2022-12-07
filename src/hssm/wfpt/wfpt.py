"""
This file includes the WFPT class factory that produces WFPT classes that support
arbitrary log-likelihood functions and random number generation functions, and
provides utility functions for handling LANs.
"""

from __future__ import annotations

from os import PathLike
from typing import Callable, List, Type

import aesara.tensor as at
import onnx
import pymc as pm
from aesara.tensor.random.op import RandomVariable
from numpy.typing import ArrayLike

from .classic import WFPTClassic, WFPTRandomVariable
from .lan import LAN

LogLikeFunc = Callable[..., ArrayLike]
LogLikeGrad = Callable[..., ArrayLike]


class WFPT:
    """
    This is a class factory for producing for Wiener First-Passage Time (WFPT)
    distributionsthat supports arbitrary log-likelihood functions and random number
    generation ops.
    """

    @classmethod
    def make_distribution(
        cls,
        loglik: LogLikeFunc | None,
        rv: Type[RandomVariable] | None,
        list_params: List[str] | None,
    ) -> Type[pm.Distribution]:

        if loglik is None:
            return WFPTClassic

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
                    at.as_tensor_variable(pm.floatX(kwargs[param]))
                    for param in cls.params
                ]
                other_kwargs = {k: v for k, v in kwargs.items() if k not in cls.params}
                return super().dist(dist_params, **other_kwargs)

            def logp(data, *dist_params):  # pylint: disable=E0213

                return loglik(data, *dist_params)

        return WFPTDistribution

    @classmethod
    def make_classic_distribution(cls) -> Type[WFPTClassic]:
        """Returns the classic WFPT Distribution."""

        return WFPTClassic

    @classmethod
    def make_lan_distribution(
        cls,
        model: str | PathLike | onnx.model,
        list_params: List[str],
        rv: Type[RandomVariable] | None = None,
        backend: str | None = "aesara",
        compile_funcs: bool = True,
    ) -> Type[pm.Distribution]:
        """Produces a PyMC distribution that uses the provided ONNX model as
        its log-likelihood function.

        Args:
            model: The path of the ONNX model, or one already loaded in memory.
            backend: Whether to use "aesara" or "jax" as the backend of the
                log-likelihood computation. If `jax`, the function will be wrapped in an
                aesara Op.
            list_params: A list of the names of the parameters following the order of
                how they are fed to the LAN.
            rv: The RandomVariable Op used for posterior sampling.
            compile_funcs: Whether the JAX functions should be compiled, if the backend
                is set to JAX.
        Returns:
            A PyMC Distribution class that uses the ONNX model as its log-likelihood
            function.
        """
        if isinstance(model, (str, PathLike)):
            model = onnx.load(model)

        lan_logp = None

        if backend == "aesara":
            lan_logp = LAN.make_aesara_logp(model)
        elif backend == "jax":
            logp, logp_grad, logp_nojit = LAN.make_jax_logp_funcs_from_onnx(
                model,
                n_params=len(list_params),
                compile_funcs=compile_funcs,
            )
            lan_logp = LAN.make_jax_logp_ops(logp, logp_grad, logp_nojit)

        return cls.make_distribution(lan_logp, rv, list_params)
