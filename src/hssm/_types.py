"""Type definitions for the HSSM package."""

from os import PathLike
from typing import Any, Callable, Literal, NotRequired, Optional, TypedDict, Union

import bambi as bmb
import numpy as np
from pymc import Distribution
from pytensor.graph.op import Op

LogLik = Union[str, PathLike, Callable, Op, type[Distribution]]
ParamSpec = Union[float, dict[str, Any], bmb.Prior, None]

SupportedModels = Literal[
    "ddm",
    "ddm_sdv",
    "full_ddm",
    "angle",
    "levy",
    "ornstein",
    "weibull",
    "race_no_bias_angle_4",
    "ddm_seq2_no_bias",
    "gamma_drift",
    "lba3",
    "lba4",
    "lba2",
    "racing_diffusion_3",
    "poisson_race",
    "softmax_inv_temperature_2",
    "softmax_inv_temperature_3",
]


LoglikKind = Literal["analytical", "approx_differentiable", "blackbox"]


class LoglikConfig(TypedDict):
    """Type for the value of LoglikConfig."""

    loglik: LogLik
    backend: Optional[Literal["jax", "pytensor"]]
    default_priors: dict[str, ParamSpec]
    bounds: dict[str, tuple[float, float]]
    extra_fields: Optional[list[str]]


LoglikConfigs = dict[LoglikKind, LoglikConfig]


class ResponseDomainSpec(TypedDict):
    """Canonical metadata for one physical response column."""

    kind: Literal["categorical", "continuous", "circular"]
    values: NotRequired[tuple[int, ...]]
    bounds: NotRequired[tuple[float, float]]


class DefaultConfig(TypedDict):
    """Type for the value of DefaultConfig."""

    response: list[str]
    list_params: list[str]
    choices: NotRequired[list[int]]
    response_domains: NotRequired[dict[str, ResponseDomainSpec]]
    description: Optional[str]
    likelihoods: LoglikConfigs


DefaultConfigs = dict[SupportedModels, DefaultConfig]

LogLikeFunc = Callable[..., np.ndarray]
LogLikeGrad = Callable[..., np.ndarray]
