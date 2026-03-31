"""Config and ModelConfig classes that process configs."""

# This is necessary to enable forward looking
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union, cast, get_args

from bambi import Prior

from ._types import LogLik, LoglikKind, SupportedModels
from .defaults import (
    default_model_config,
)
from .modelconfig import get_default_model_config
from .register import register_model

if TYPE_CHECKING:
    from pytensor.tensor.random.op import RandomVariable

import logging

from ssms.config import model_config as ssms_model_config

_logger = logging.getLogger("hssm")


# ====== Centralized SSM defaults =====
DEFAULT_SSM_OBSERVED_DATA = ["rt", "response"]
DEFAULT_SSM_CHOICES = (0, 1)

ParamSpec = Union[float, dict[str, Any], Prior, None]


@dataclass
class BaseModelConfig(ABC):
    """Base configuration class for all model types."""

    # Core identification
    model_name: str
    description: str | None = None

    # Data specification
    response: list[str] | None = field(default_factory=DEFAULT_SSM_OBSERVED_DATA.copy)
    choices: tuple[int, ...] | None = DEFAULT_SSM_CHOICES

    # Parameter specification
    list_params: list[str] | None = None
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Likelihood configuration
    loglik: LogLik | None = None
    loglik_kind: LoglikKind | None = None
    backend: Literal["jax", "pytensor"] | None = None

    # Additional data requirements
    extra_fields: list[str] | None = None

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration. Must be implemented by subclasses."""
        ...

    @abstractmethod
    def get_defaults(self, param: str) -> Any:
        """Get default values for a parameter. Must be implemented by subclasses."""
        ...

    @property
    def n_params(self) -> int | None:
        """Return the number of parameters."""
        return len(self.list_params) if self.list_params else None

    @property
    def n_extra_fields(self) -> int | None:
        """Return the number of extra fields."""
        return len(self.extra_fields) if self.extra_fields else None


@dataclass
class Config(BaseModelConfig):
    """Config class that stores the configurations for models."""

    rv: RandomVariable | None = None
    # Fields with dictionaries are automatically deepcopied
    default_priors: dict[str, ParamSpec] = field(default_factory=dict)

    def __post_init__(self):
        """Validate that loglik_kind is provided."""
        if self.loglik_kind is None:
            raise ValueError("loglik_kind is required for Config")

    @classmethod
    def from_defaults(
        cls, model_name: SupportedModels | str, loglik_kind: LoglikKind | None
    ):
        """Generate a Config object from defaults.

        Parameters
        ----------
        model_name
            The name of the model.
        loglik_kind
            The kind of the log-likelihood for the model.
        """
        model_name_casted = cast("SupportedModels", model_name)
        if all(
            [
                model_name_casted in get_args(SupportedModels),
                model_name_casted not in default_model_config,
            ]
        ):
            register_model(
                model_name_casted, **get_default_model_config(model_name_casted)
            )

        if loglik_kind is None:
            if model_name not in default_model_config:
                raise ValueError(
                    "When using a custom model, please provide a `loglik_kind.`"
                )
            # Setting loglik_kind to be the first of analytical or
            # approx_differentiable
            for kind in ["analytical", "approx_differentiable", "blackbox"]:
                model_name = cast("SupportedModels", model_name)
                default_config = deepcopy(default_model_config[model_name])
                if kind in default_config["likelihoods"]:
                    kind = cast("LoglikKind", kind)
                    loglik_config = default_config["likelihoods"][kind]

                    return Config(
                        model_name=model_name,
                        loglik_kind=kind,
                        response=list(default_config["response"]),
                        choices=tuple(default_config["choices"]),
                        list_params=default_config["list_params"],
                        description=default_config["description"],
                        **loglik_config,
                    )

            raise ValueError(
                "No default model_config is found. Please provide a `loglik_kind."
            )
        else:
            if loglik_kind not in [
                "analytical",
                "approx_differentiable",
                "blackbox",
            ]:
                raise ValueError(
                    "`loglik_kind`, when provided, must be one of "
                    + '"analytical", "approx_differentiable", "blackbox".'
                )
            if model_name in default_model_config:
                model_name = cast("SupportedModels", model_name)
                default_config = deepcopy(default_model_config[model_name])
                if loglik_kind in default_config["likelihoods"]:
                    loglik_config = default_config["likelihoods"][loglik_kind]
                    return Config(
                        model_name=model_name,
                        loglik_kind=loglik_kind,
                        response=list(default_config["response"]),
                        choices=tuple(default_config["choices"]),
                        list_params=default_config["list_params"],
                        description=default_config["description"],
                        **loglik_config,
                    )
                return Config(
                    model_name=model_name,
                    loglik_kind=loglik_kind,
                    response=list(default_config["response"]),
                    choices=tuple(default_config["choices"]),
                    list_params=default_config["list_params"],
                    description=default_config["description"],
                )

            return Config(
                model_name=model_name,
                loglik_kind=loglik_kind,
                response=DEFAULT_SSM_OBSERVED_DATA,
            )

    def update_loglik(self, loglik: Any | None) -> None:
        """Update the log-likelihood function from user input.

        Parameters
        ----------
        loglik : optional
            A user-defined log-likelihood function.
        """
        if loglik is None:
            return

        self.loglik = loglik

    def update_choices(self, choices: tuple[int, ...] | None) -> None:
        """Update the choices from user input.

        Parameters
        ----------
        choices : tuple[int, ...] | None
            A tuple of choices.
        """
        if choices is None:
            return

        self.choices = choices

    def update_config(self, user_config: ModelConfig) -> None:
        """Update the object from a ModelConfig object.

        Parameters
        ----------
        user_config: ModelConfig
            User specified ModelConfig used update self.
        """
        if user_config.response is not None:
            self.response = list(user_config.response)  # type: ignore[assignment]
        if user_config.list_params is not None:
            self.list_params = user_config.list_params
        if user_config.choices is not None:
            self.choices = user_config.choices
        if user_config.rv is not None:
            self.rv = user_config.rv

        if (
            self.loglik_kind == "approx_differentiable"
            and user_config.backend is not None
        ):
            self.backend = user_config.backend

        self.default_priors |= user_config.default_priors
        self.bounds |= user_config.bounds
        self.extra_fields = user_config.extra_fields

    def validate(self) -> None:
        """Ensure that mandatory fields are not None."""
        if self.response is None:
            raise ValueError("Please provide `response` columns in the configuration.")
        if self.list_params is None:
            raise ValueError("Please provide `list_params`.")
        if self.choices is None:
            raise ValueError("Please provide `choices`.")
        if self.loglik is None:
            raise ValueError("Please provide a log-likelihood function via `loglik`.")
        if self.loglik_kind == "approx_differentiable" and self.backend is None:
            raise ValueError("Please provide `backend` via `model_config`.")

    def get_defaults(
        self, param: str
    ) -> tuple[ParamSpec | None, tuple[float, float] | None]:
        """Return the default prior and bounds for a parameter.

        Parameters
        ----------
        param
            The name of the parameter.
        """
        return self.default_priors.get(param), self.bounds.get(param)

    @classmethod
    def _build_model_config(
        cls,
        model: SupportedModels | str,
        loglik_kind: LoglikKind | None,
        model_config: ModelConfig | dict | None,
        choices: list[int] | tuple[int, ...] | None,
        loglik: Any = None,
    ) -> Config:
        """Build and return a validated Config for standard HSSM models.

        Resolves defaults, normalizes dict/ModelConfig overrides, applies
        choices and loglik precedence rules, then validates before returning.
        """
        config = cls.from_defaults(model, loglik_kind)

        if model_config is not None:
            final_config = _normalize_model_config_with_choices(model_config, choices)
            config.update_config(final_config)

        # No model_config provided: apply `choices` when appropriate.
        # If caller passed a SupportedModels string, ignore explicit `choices`.
        if model in get_args(SupportedModels) and choices is not None:
            _logger.info(
                "Model string is in SupportedModels. Ignoring choices arguments."
            )

        # If model is not a supported built-in, prefer explicit choices or
        # fall back to ssms-simulators lookup when available.
        if model not in get_args(SupportedModels):
            if choices is not None:
                config.update_choices(choices)
            elif model in ssms_model_config:
                config.update_choices(ssms_model_config[model]["choices"])
                _logger.info(
                    "choices argument passed as None, "
                    "but found %s in ssms-simulators. "
                    "Using choices, from ssm-simulators configs: %s",
                    model,
                    ssms_model_config[model]["choices"],
                )

        config.update_loglik(loglik)
        config.validate()
        return config


@dataclass
class ModelConfig:
    """Representation for model_config provided by the user."""

    response: tuple[str, ...] | None = None
    list_params: list[str] | None = None
    choices: tuple[int, ...] | None = None
    default_priors: dict[str, ParamSpec] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    backend: Literal["jax", "pytensor"] | None = None
    rv: RandomVariable | None = None
    extra_fields: list[str] | None = None


def _normalize_model_config_with_choices(
    model_config: "ModelConfig" | dict[str, Any],
    choices: list[int] | tuple[int, ...] | None,
) -> "ModelConfig":
    """Normalize a user-supplied model_config and apply choices.

    Returns a fresh :class:`ModelConfig` instance and does not mutate the
    caller's objects. If both ``model_config`` and ``choices`` are provided
    and ``model_config`` already contains ``choices``, the value from
    ``model_config`` wins (and a log entry is emitted).
    """
    # Normalize input to a mutable dict so we can coerce and avoid mutating
    # the caller's objects. Build a fresh ModelConfig from that dict.
    if isinstance(model_config, ModelConfig):
        mc: dict[str, Any] = {
            k: getattr(model_config, k) for k in model_config.__dataclass_fields__
        }
    else:
        mc = model_config.copy()

    # Coerce any existing choices on the input to a tuple for immutability
    if mc.get("choices") is not None:
        mc["choices"] = tuple(mc["choices"])

    # If caller didn't provide an explicit `choices` argument, return the
    # normalized ModelConfig built from the input (fresh instance).
    if choices is None:
        return ModelConfig(**{k: v for k, v in mc.items() if v is not None})

    # Caller provided choices; prefer the one embedded in model_config if
    # present, otherwise apply the provided value (coerced to tuple).
    if mc.get("choices") is not None:
        _logger.info(
            "choices list provided in both model_config and "
            "as an argument directly. Using the one provided in "
            "model_config. We recommend providing choices in model_config."
        )
        return ModelConfig(**{k: v for k, v in mc.items() if v is not None})

    mc["choices"] = tuple(choices)
    return ModelConfig(**{k: v for k, v in mc.items() if v is not None})
