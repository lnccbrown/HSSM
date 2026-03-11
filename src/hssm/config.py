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

# ====== Centralized RLSSM defaults =====
DEFAULT_SSM_OBSERVED_DATA = ["rt", "response"]
DEFAULT_RLSSM_OBSERVED_DATA = ["rt", "response"]
DEFAULT_SSM_CHOICES = (0, 1)

RLSSM_REQUIRED_FIELDS = (
    "model_name",
    "description",
    "list_params",
    "bounds",
    "params_default",
    "choices",
    "decision_process",
    "learning_process",
    "response",
    "decision_process_loglik_kind",
    "learning_process_loglik_kind",
    "extra_fields",
    "ssm_logp_func",
)

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
                response=DEFAULT_RLSSM_OBSERVED_DATA,
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
        choices : tuple[int, ...]
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
        model: "SupportedModels | str",
        loglik_kind: "LoglikKind | None",
        model_config: "ModelConfig | dict | None",
        choices: "list[int] | None",
        loglik: Any = None,
    ) -> "Config":
        """Build and return a validated Config for standard HSSM models.

        Resolves defaults, normalizes dict/ModelConfig overrides, applies
        choices and loglik precedence rules, then validates before returning.
        """
        config = cls.from_defaults(model, loglik_kind)

        if model_config is not None:
            has_choices = (
                isinstance(model_config, dict)
                and "choices" in model_config
                or isinstance(model_config, ModelConfig)
                and model_config.choices is not None
            )
            if choices is not None:
                if has_choices:
                    _logger.info(
                        "choices list provided in both model_config and "
                        "as an argument directly."
                        " Using the one provided in model_config. \n"
                        "We recommend providing choices in model_config."
                    )
                else:
                    if isinstance(model_config, dict):
                        model_config = {**model_config, "choices": choices}
                    else:
                        model_config_dict = {
                            k: getattr(model_config, k)
                            for k in model_config.__dataclass_fields__
                            if getattr(model_config, k) is not None
                        }
                        model_config_dict["choices"] = choices
                        model_config = model_config_dict

            final_config = (
                model_config
                if isinstance(model_config, ModelConfig)
                else ModelConfig(**model_config)
            )
            config.update_config(final_config)

        else:
            if model in get_args(SupportedModels):
                if choices is not None:
                    _logger.info(
                        "Model string is in SupportedModels."
                        " Ignoring choices arguments."
                    )
            else:
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
class RLSSMConfig(BaseModelConfig):
    """Config for reinforcement learning + sequential sampling models.

    This configuration class is designed for models that combine reinforcement
    learning processes with sequential sampling decision models (RLSSM).

    The ``ssm_logp_func`` field holds the fully annotated JAX SSM log-likelihood
    function (an :class:`AnnotatedFunction`) that is passed directly to
    ``make_rl_logp_op``.  It supersedes the ``loglik`` / ``loglik_kind`` workflow
    used by :class:`HSSM`: the Op is built from ``ssm_logp_func`` and therefore
    no ``loglik`` callable needs to be provided.
    """

    decision_process_loglik_kind: str = field(kw_only=True)
    learning_process_loglik_kind: str = field(kw_only=True)
    params_default: list[float] = field(kw_only=True)
    decision_process: str | ModelConfig = field(kw_only=True)
    learning_process: dict[str, Any] = field(kw_only=True)
    # The fully annotated SSM log-likelihood function used by make_rl_logp_op.
    # Type is Any to avoid a hard dependency on the AnnotatedFunction Protocol at
    # import time; validated at runtime in validate().
    ssm_logp_func: Any = field(default=None, kw_only=True)

    def __post_init__(self):
        """Set default loglik_kind for RLSSM models if not provided."""
        if self.loglik_kind is None:
            self.loglik_kind = "approx_differentiable"

    @classmethod
    def from_defaults(
        cls, model_name: SupportedModels | str, loglik_kind: LoglikKind | None
    ) -> Config:
        """Return the shared Config defaults (delegated to :class:`Config`)."""
        return Config.from_defaults(model_name, loglik_kind)

    @classmethod
    def from_rlssm_dict(cls, config_dict: dict[str, Any]) -> "RLSSMConfig":
        """
        Create RLSSMConfig from a configuration dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Dictionary containing model configuration. Expected keys:
                - model_name: Model identifier (required)
                - description: Model description (required)
                - list_params: List of parameter names (required)
                - extra_fields: List of extra field names from data (required)
                - params_default: Default parameter values (required)
                - bounds: Parameter bounds (required)
                - response: Response column names (required)
                - choices: Valid choice values (required)
                - decision_process: Decision process specification (required)
                - learning_process: Learning process functions (required)
                - decision_process_loglik_kind: Likelihood kind for decision process
                  (required)
                - learning_process_loglik_kind: Likelihood kind for learning process
                  (required)

        Returns
        -------
        RLSSMConfig
            Configured RLSSM model configuration object.
        """
        # Check for required fields and raise explicit errors if missing
        for field_name in RLSSM_REQUIRED_FIELDS:
            if field_name not in config_dict or config_dict[field_name] is None:
                raise ValueError(f"{field_name} must be provided in config_dict")

        return cls(
            model_name=config_dict["model_name"],
            description=config_dict["description"],
            list_params=config_dict["list_params"],
            extra_fields=config_dict.get("extra_fields"),
            params_default=config_dict["params_default"],
            decision_process=config_dict["decision_process"],
            learning_process=config_dict["learning_process"],
            ssm_logp_func=config_dict["ssm_logp_func"],
            bounds=config_dict.get("bounds", {}),
            response=config_dict["response"],
            choices=config_dict["choices"],
            decision_process_loglik_kind=config_dict["decision_process_loglik_kind"],
            learning_process_loglik_kind=config_dict["learning_process_loglik_kind"],
        )

    def validate(self) -> None:
        """Validate RLSSM configuration.

        Raises
        ------
        ValueError
            If required fields are missing or inconsistent.
        """
        if self.response is None:
            raise ValueError("Please provide `response` columns in the configuration.")
        if self.list_params is None:
            raise ValueError("Please provide `list_params` in the configuration.")
        if self.choices is None:
            raise ValueError("Please provide `choices` in the configuration.")
        if self.decision_process is None:
            raise ValueError("Please specify a `decision_process`.")
        if self.ssm_logp_func is None:
            raise ValueError(
                "Please provide `ssm_logp_func`: the fully annotated JAX SSM "
                "log-likelihood function required by `make_rl_logp_op`."
            )
        if not callable(self.ssm_logp_func):
            raise ValueError(
                "`ssm_logp_func` must be a callable, "
                f"but got {type(self.ssm_logp_func)!r}."
            )
        missing_attrs = [
            attr
            for attr in ("inputs", "outputs", "computed")
            if not hasattr(self.ssm_logp_func, attr)
        ]
        if missing_attrs:
            raise ValueError(
                "`ssm_logp_func` must be decorated with `@annotate_function` "
                "so that it carries the attributes required by `make_rl_logp_op`. "
                f"Missing attribute(s): {missing_attrs}. "
                "Decorate the function like:\n\n"
                "    @annotate_function(\n"
                "        inputs=[...], outputs=[...], computed={...}\n"
                "    )\n"
                "    def my_ssm_logp(lan_matrix): ..."
            )

        # Validate parameter defaults consistency
        if self.params_default and self.list_params:
            if len(self.params_default) != len(self.list_params):
                raise ValueError(
                    f"params_default length ({len(self.params_default)}) doesn't "
                    f"match list_params length ({len(self.list_params)})"
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

    def to_config(self) -> "Config":
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
        # Validate parameter defaults consistency before conversion
        if self.params_default and self.list_params:
            if len(self.params_default) != len(self.list_params):
                raise ValueError(
                    f"params_default length ({len(self.params_default)}) doesn't "
                    f"match list_params length ({len(self.list_params)}). "
                    "This would result in silent data loss during conversion."
                )

        # Transform params_default list to default_priors dict
        default_priors = (
            {
                param: default
                for param, default in zip(self.list_params, self.params_default)
            }
            if self.list_params and self.params_default
            else {}
        )

        return Config(
            model_name=self.model_name,
            loglik_kind=self.loglik_kind,
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

    def to_model_config(self) -> "ModelConfig":
        """Build a :class:`ModelConfig` from this :class:`RLSSMConfig`.

        All fields are sourced from ``self``; the backend is fixed to ``"jax"``
        because RLSSM exclusively uses the JAX backend.

        ``default_priors`` is intentionally left empty so the
        ``prior_settings="safe"`` mechanism in :class:`~hssm.base.HSSMBase`
        assigns sensible priors from bounds rather than fixing every parameter
        to a constant scalar.
        """
        return ModelConfig(
            response=tuple(self.response),  # type: ignore[arg-type]
            list_params=list(self.list_params),  # type: ignore[arg-type]
            choices=tuple(self.choices),  # type: ignore[arg-type]
            default_priors={},
            bounds=self.bounds,
            extra_fields=self.extra_fields,
            backend="jax",
        )

    def _build_model_config(self, loglik_op: Any) -> "Config":
        """Build a validated :class:`Config` for use by :class:`~hssm.rl.rlssm.RLSSM`.

        Converts this :class:`RLSSMConfig` to a :class:`ModelConfig`, then
        delegates to :meth:`Config._build_model_config` using the pre-built
        differentiable Op as ``loglik``.

        Parameters
        ----------
        loglik_op
            The differentiable pytensor Op produced by
            :func:`~hssm.rl.likelihoods.builder.make_rl_logp_op`.

        Returns
        -------
        Config
            A fully validated :class:`Config` ready to pass to
            :meth:`~hssm.base.HSSMBase.__init__`.
        """
        mc = self.to_model_config()
        return Config._build_model_config(
            self.model_name,
            "approx_differentiable",
            mc,
            None,
            loglik_op,
        )


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
