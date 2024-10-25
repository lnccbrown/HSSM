"""Config and ModelConfig classes that process configs."""

# This is necessary to enable forward looking
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Type, Union, cast

import bambi as bmb

from .defaults import LogLik, LoglikKind, SupportedModels, default_model_config
from .param.user_param import (
    to_dict_shallow,
)
from .param.utils import SerializedPrior, deserialize_prior, serialize_prior

if TYPE_CHECKING:
    from os import PathLike

    import pymc as pm
    import pytensor
    from pytensor.tensor.random.op import RandomVariable


ParamSpec = Union[float, dict[str, Any], bmb.Prior, None]

_logger = logging.getLogger("hssm")


@dataclass(slots=True)
class Config:
    """Config class that stores the configurations for models."""

    model_name: SupportedModels | str
    loglik_kind: LoglikKind
    response: list[str] | None = None
    choices: list[int] | None = None
    list_params: list[str] | None = None
    description: str | None = None
    loglik: LogLik | None = None
    backend: Literal["jax", "pytensor"] | None = None
    rv: RandomVariable | None = None
    extra_fields: list[str] | None = None
    # Fields with dictionaries are automatically deepcopied
    default_priors: dict[str, ParamSpec] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    user_config: ModelConfig | None = field(init=False, default=None)

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
        if loglik_kind is None:
            if model_name not in default_model_config:
                raise ValueError(
                    "When using a custom model, please provide a `loglik_kind.`"
                )
            # Setting loglik_kind to be the first of analytical or
            # approx_differentiable
            for kind in ["analytical", "approx_differentiable", "blackbox"]:
                model_name = cast(SupportedModels, model_name)
                default_config = deepcopy(default_model_config[model_name])
                if kind in default_config["likelihoods"]:
                    kind = cast(LoglikKind, kind)
                    loglik_config = default_config["likelihoods"][kind]

                    return Config(
                        model_name,
                        loglik_kind=kind,
                        response=default_config["response"],
                        choices=default_config["choices"],
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
                model_name = cast(SupportedModels, model_name)
                default_config = deepcopy(default_model_config[model_name])
                if loglik_kind in default_config["likelihoods"]:
                    loglik_config = default_config["likelihoods"][loglik_kind]
                    return Config(
                        model_name,
                        loglik_kind=loglik_kind,
                        response=default_config["response"],
                        choices=default_config["choices"],
                        list_params=default_config["list_params"],
                        description=default_config["description"],
                        **loglik_config,
                    )
                return Config(
                    model_name,
                    loglik_kind=loglik_kind,
                    response=default_config["response"],
                    choices=default_config["choices"],
                    list_params=default_config["list_params"],
                    description=default_config["description"],
                )

            return Config(model_name, loglik_kind, response=["rt", "response"])

    def update_config(
        self,
        user_config: ModelConfig | None = None,
        choices: list[int] | None = None,
        loglik: str
        | PathLike
        | pytensor.graph.Op
        | Type[pm.Distribution]
        | None = None,
    ) -> None:
        """Update the object from a ModelConfig object.

        Parameters
        ----------
        user_config
            User specified ModelConfig used update self.
        choices
            The choices list for the model.
        loglik
            The log-likelihood function.
        """
        if user_config is not None:
            user_config = (
                ModelConfig(**user_config)
                if isinstance(user_config, dict)
                else user_config
            )
            if choices is not None:
                if hasattr(user_config, "choices"):
                    _logger.warning(
                        "choices list provided in both model_config and "
                        "as an argument directly."
                        " Using the one provided in model_config. \n"
                        "We recommend providing choices in model_config."
                    )
                else:
                    self.choices = choices

            if user_config.response is not None:
                self.response = user_config.response
            if user_config.list_params is not None:
                self.list_params = user_config.list_params
            if user_config.choices is not None:
                self.choices = user_config.choices

            if (
                self.loglik_kind == "approx_differentiable"
                and user_config.backend is not None
            ):
                self.backend = user_config.backend

            self.default_priors |= user_config.default_priors
            self.bounds |= user_config.bounds
            self.extra_fields = user_config.extra_fields
        else:
            # This is to allow users to override default choices.
            # For some models, such as `ddm` and custom likelihoods,
            # This can be useful.
            if choices is not None:
                self.choices = choices

        if loglik is not None:
            self.loglik = loglik
        self.validate()

    def validate(self) -> None:
        """Ensure that mandatory fields are not None."""
        if self.response is None:
            raise ValueError("Please provide `response` via `model_config`.")
        if self.list_params is None:
            raise ValueError("Please provide `list_params` via `model_config`.")
        if self.choices is None:
            raise ValueError("Please provide `choices` via `model_config`.")
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


@dataclass(slots=True)
class ModelConfig:
    """Representation for model_config provided by the user."""

    response: list[str] | None = None
    list_params: list[str] | None = None
    choices: list[int] | None = None
    default_priors: dict[str, ParamSpec] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    backend: Literal["jax", "pytensor"] | None = None
    rv: RandomVariable | None = None
    extra_fields: list[str] | None = None

    def serialize(self) -> dict[str, SerializedPrior]:
        """Serialize the object to a dictionary."""
        result = to_dict_shallow(self)

        if self.rv is not None:
            raise ValueError("Cannot serialize RandomVariable object.")

        if "default_priors" in result:
            result["default_priors"] = {
                key: serialize_prior(value)
                for key, value in self.default_priors.items()
            }

        return result

    @classmethod
    def deserialize(cls, d: dict[str, Any]) -> ModelConfig:
        """Deserialize a serialized model_config.

        Parameters
        ----------
        d
            A dictionary of serialized model_config.

        Returns
        -------
        ModelConfig
            The deserialized model_config.
        """
        if "default_priors" in d:
            d = d.copy()
            d["default_priors"] = {
                key: deserialize_prior(value)
                for key, value in d["default_priors"].items()
            }

        return cls(**d)
