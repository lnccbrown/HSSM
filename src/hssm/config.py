"""Config and ModelConfig classes that process configs."""

# This is necessary to enable forward looking
from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Union, cast, get_args

import bambi as bmb

from ._types import LogLik, LoglikKind, SupportedModels
from .defaults import (
    default_model_config,
)
from .modelconfig import get_default_model_config
from .register import register_model

if TYPE_CHECKING:
    from pytensor.tensor.random.op import RandomVariable


ParamSpec = Union[float, dict[str, Any], bmb.Prior, None]


@dataclass
class BaseModelConfig(ABC):
    """Base configuration class for all model types."""

    # Core identification
    model_name: str
    description: str | None = None

    # Data specification
    response: list[str] | None = None
    choices: list[int] | None = None

    # Parameter specification
    list_params: list[str] | None = None
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Likelihood configuration
    loglik: LogLik | None = None
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


@dataclass
class Config(BaseModelConfig):
    """Config class that stores the configurations for models."""

    loglik_kind: LoglikKind = field(default=None)  # type: ignore
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
                model_name = cast("SupportedModels", model_name)
                default_config = deepcopy(default_model_config[model_name])
                if loglik_kind in default_config["likelihoods"]:
                    loglik_config = default_config["likelihoods"][loglik_kind]
                    return Config(
                        model_name=model_name,
                        loglik_kind=loglik_kind,
                        response=default_config["response"],
                        choices=default_config["choices"],
                        list_params=default_config["list_params"],
                        description=default_config["description"],
                        **loglik_config,
                    )
                return Config(
                    model_name=model_name,
                    loglik_kind=loglik_kind,
                    response=default_config["response"],
                    choices=default_config["choices"],
                    list_params=default_config["list_params"],
                    description=default_config["description"],
                )

            return Config(
                model_name=model_name,
                loglik_kind=loglik_kind,
                response=["rt", "response"],
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

    def update_choices(self, choices: list[int] | None) -> None:
        """Update the choices from user input.

        Parameters
        ----------
        choices : list[int]
            A list of choices.
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
            self.response = user_config.response
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
            raise ValueError(
                "Please provide `response` columns (e.g., via the `response` or `data` field)."
            )
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


@dataclass
class RLSSMConfig(BaseModelConfig):
    """Config for reinforcement learning + sequential sampling models.

    This configuration class is designed for models that combine reinforcement
    learning processes with sequential sampling decision models (RLSSM).
    """

    # RLSSM-specific: parameter defaults as list (matches list_params order)
    params_default: list[float] = field(default_factory=list)

    # RLSSM-specific: hierarchical structure
    decision_process: str | ModelConfig | None = None
    learning_process: dict[str, Any] = field(default_factory=dict)

    # Number of parameters (for validation)
    n_params: int | None = None
    n_extra_fields: int | None = None

    # Additional metadata for RLSSM models
    decision_model: str | None = None  # e.g., "LAN"
    lan_model: str | None = None  # e.g., "angle", "dev_lba_angle_3_v2"

    @classmethod
    def from_rlssm_dict(cls, model_name: str, config_dict: dict[str, Any]):
        """Create RLSSMConfig from rlssm_model_config_list style dictionary.

        Parameters
        ----------
        model_name
            The name of the RLSSM model.
        config_dict
            Dictionary containing model configuration. Expected keys:
            - name: Model name
            - description: Model description
            - n_params: Number of parameters
            - n_extra_fields: Number of extra data fields
            - list_params: List of parameter names
            - extra_fields: List of extra field names from data
            - decision_model: Name of decision model (e.g., "LAN")
            - LAN: Specific LAN model variant
            - params_default (optional): Default parameter values
            - bounds (optional): Parameter bounds
            - response (optional): Response column names
            - data (optional): Alias for response column names
            - choices (optional): Valid choice values
            - learning_process (optional): Learning process functions

        Returns
        -------
        RLSSMConfig
            Configured RLSSM model configuration object.
        """
        # Support both 'response' and 'data' keys, with 'data' taking precedence
        response_value = config_dict.get(
            "data", config_dict.get("response", ["rt", "response"])
        )

        return cls(
            model_name=model_name,
            description=config_dict.get("description"),
            list_params=config_dict.get("list_params"),
            extra_fields=config_dict.get("extra_fields"),
            params_default=config_dict.get("params_default", []),
            n_params=config_dict.get("n_params"),
            n_extra_fields=config_dict.get("n_extra_fields"),
            decision_process=config_dict.get("decision_model"),
            decision_model=config_dict.get("decision_model"),
            lan_model=config_dict.get("LAN"),
            learning_process=config_dict.get("learning_process", {}),
            bounds=config_dict.get("bounds", {}),
            response=response_value,
            choices=config_dict.get("choices", [0, 1]),
        )

    def validate(self) -> None:
        """Validate RLSSM configuration.

        Raises
        ------
        ValueError
            If required fields are missing or inconsistent.
        """
        if self.response is None:
            raise ValueError("Please provide `response` columns via `model_config`.")
        if self.list_params is None:
            raise ValueError("Please provide `list_params` via `model_config`.")
        if self.choices is None:
            raise ValueError("Please provide `choices` via `model_config`.")
        if self.decision_process is None:
            raise ValueError("Please specify a `decision_process`.")

        # Validate parameter count consistency
        if self.n_params is not None and len(self.list_params) != self.n_params:
            raise ValueError(
                f"list_params length ({len(self.list_params)}) doesn't match "
                f"n_params ({self.n_params})"
            )

        if self.params_default and len(self.params_default) != len(self.list_params):
            raise ValueError(
                f"params_default length ({len(self.params_default)}) doesn't match "
                f"list_params length ({len(self.list_params)})"
            )

        # Validate extra fields consistency
        if (
            self.n_extra_fields is not None
            and self.extra_fields is not None
            and len(self.extra_fields) != self.n_extra_fields
        ):
            raise ValueError(
                f"extra_fields length ({len(self.extra_fields)}) doesn't match "
                f"n_extra_fields ({self.n_extra_fields})"
            )

    def get_defaults(
        self, param: str
    ) -> tuple[float | None, tuple[float, float] | None]:
        """Return default value and bounds for a parameter.

        Parameters
        ----------
        param
            The name of the parameter.

        Returns
        -------
        tuple
            A tuple of (default_value, bounds) where:
            - default_value is a float or None if not found
            - bounds is a tuple (lower, upper) or None if not found
        """
        # Try to find the parameter in list_params and get its default value
        default_val = None
        if self.list_params is not None:
            try:
                param_idx = self.list_params.index(param)
                if self.params_default and param_idx < len(self.params_default):
                    default_val = self.params_default[param_idx]
            except ValueError:
                # Parameter not in list_params
                pass

        return default_val, self.bounds.get(param)

    def to_config(self) -> Config:
        """Convert to standard Config for compatibility with HSSM.

        This method transforms the RLSSM configuration into a standard Config
        object that can be used with the existing HSSM infrastructure.

        Returns
        -------
        Config
            A Config object with RLSSM parameters mapped to standard format.

        Notes
        -----
        The transformation converts params_default list to default_priors dict,
        mapping parameter names to their default values.
        """
        # Transform params_default list to default_priors dict
        default_priors = {}
        if self.list_params and self.params_default:
            default_priors = {
                param: default
                for param, default in zip(self.list_params, self.params_default)
            }

        return Config(
            model_name=self.model_name,
            loglik_kind="approx_differentiable",  # RLSSM models use approx_diff
            response=self.response,
            choices=self.choices,
            list_params=self.list_params,
            description=self.description,
            bounds=self.bounds,
            default_priors=cast(
                "dict[str, float | dict[str, Any] | Any | None]", default_priors
            ),
            extra_fields=self.extra_fields,
            backend=self.backend or "jax",  # RLSSM typically uses JAX
            loglik=self.loglik,
        )

    @property
    def data(self) -> list[str] | None:
        """Alias for response field for RLSSM-style naming.

        This property allows using 'data' as an alternative name for 'response',
        which may be more intuitive in RLSSM contexts.

        Returns
        -------
        list[str] | None
            The response column names.
        """
        return self.response

    @data.setter
    def data(self, value: list[str] | None) -> None:
        """Set response field via data alias.

        Parameters
        ----------
        value
            The response column names to set.
        """
        self.response = value


@dataclass
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
