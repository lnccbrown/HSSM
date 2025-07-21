"""Type definitions for the HSSM package."""

from os import PathLike
from typing import Any, Callable, Literal, Optional, TypedDict, Union

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
    "lba3",
    "lba2",
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


class DefaultConfig(TypedDict):
    """Type for the value of DefaultConfig."""

    response: list[str]
    list_params: list[str]
    choices: list[int]
    description: Optional[str]
    likelihoods: LoglikConfigs


DefaultConfigs = dict[SupportedModels, DefaultConfig]

LogLikeFunc = Callable[..., np.ndarray]
LogLikeGrad = Callable[..., np.ndarray]
