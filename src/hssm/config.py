"""Config and ModelConfig classes that process configs."""

# This is necessary to enable forward looking
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, cast

from .defaults import LogLik, LoglikKind, SupportedModels, default_model_config

if TYPE_CHECKING:
    from pytensor.tensor.random.op import RandomVariable

    from .param import ParamSpec


@dataclass
class Config:
    """Config class that stores the configurations for models."""

    model_name: SupportedModels | str
    loglik_kind: LoglikKind
    list_params: list[str] | None = None
    description: str | None = None
    loglik: LogLik | None = None
    backend: Literal["jax", "pytensor"] | None = None
    rv: RandomVariable | None = None
    extra_fields: list[str] | None = None
    # Fields with dictionaries are automatically deepcopied
    default_priors: dict[str, ParamSpec] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

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
                        kind,
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
                        loglik_kind,
                        list_params=default_config["list_params"],
                        description=default_config["description"],
                        **loglik_config,
                    )
                return Config(
                    model_name,
                    loglik_kind,
                    list_params=default_config["list_params"],
                    description=default_config["description"],
                )

            return Config(model_name, loglik_kind)

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

    def update_config(self, user_config: ModelConfig) -> None:
        """Update the object from a ModelConfig object.

        Parameters
        ----------
        loglik : optional
            A user-defined log-likelihood function.
        """
        if user_config.list_params is not None:
            self.list_params = user_config.list_params

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
        if self.list_params is None:
            raise ValueError("Please provide `list_params` via `model_config`.")
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


@dataclass
class ModelConfig:
    """Representation for model_config provided by the user."""

    list_params: list[str] | None = None
    default_priors: dict[str, ParamSpec] = field(default_factory=dict)
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    backend: Literal["jax", "pytensor"] | None = None
    rv: RandomVariable | None = None
    extra_fields: list[str] | None = None
