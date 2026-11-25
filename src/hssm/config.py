"""Config and ModelConfig classes that process configs."""

# This is necessary to enable forward looking
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from numbers import Integral, Real
from os import PathLike
from typing import TYPE_CHECKING, Any, Literal, Union, cast, get_args

import numpy as np
from bambi import Prior

from ._types import (
    LogLik,
    LoglikKind,
    NDTEdgeShift,
    ResponseDomainSpec,
    SupportedModels,
)
from .defaults import (
    default_model_config,
)
from .modelconfig import get_default_model_config
from .register import register_model

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytensor.tensor.random.op import RandomVariable

import logging

from ssms.config import get_model_registry

_logger = logging.getLogger("hssm")


# ====== Centralized SSM defaults =====
DEFAULT_SSM_OBSERVED_DATA = ["rt", "response"]
DEFAULT_SSM_CHOICES = (0, 1)

ParamSpec = Union[float, dict[str, Any], Prior, None]


def _get_ssms_model_config(model: str) -> dict[str, Any] | None:
    """Return ssm-simulators' config for ``model``, or None if ssms does not know it.

    Looks the name up in ssms' model registry rather than the
    ``ssms.config.model_config`` snapshot taken at import, so models registered
    with ``ssms.config.register_model_config`` are found too.
    """
    registry = get_model_registry()
    return registry.get(model) if registry.has_model(model) else None


def _bounds_from_ssms(ssms_config: dict[str, Any]) -> dict[str, tuple[float, float]]:
    """Convert an ssms config's ``param_bounds`` to HSSM's ``bounds`` dict.

    ssms gives bounds either as ``[[lower, ...], [upper, ...]]`` in ``params``
    order or as a ``{name: (lower, upper)}`` dict; configs built with
    ``ModelConfigBuilder`` also carry the dict form as ``param_bounds_dict``.
    """
    bounds = ssms_config.get("param_bounds_dict")
    if bounds is None:
        bounds = ssms_config.get("param_bounds")
    if bounds is None:
        return {}
    if not isinstance(bounds, dict):
        lower, upper = bounds
        bounds = dict(zip(ssms_config["params"], zip(lower, upper)))
    # A bound that names another parameter (full_ddm_rv's st < t) has no HSSM
    # form, so that parameter gets none.
    return {
        name: (float(lo), float(hi))
        for name, (lo, hi) in bounds.items()
        if not isinstance(lo, str) and not isinstance(hi, str)
    }


def _validate_ndt_edge_shift(
    ndt_edge_shift: NDTEdgeShift | None, list_params: list[str]
) -> None:
    """Check an ``ndt_edge_shift`` declaration against a likelihood's parameters.

    Shared by every config that carries the declaration (``Config`` and the RL
    decision-process registry), so the rules and messages stay the same.

    Parameters
    ----------
    ndt_edge_shift
        The declaration to check. ``None`` means nothing is declared and is
        accepted as is.
    list_params
        The parameters of the likelihood the declaration belongs to.

    Raises
    ------
    ValueError
        If the declaration is not a dict with exactly the keys ``param`` and
        ``scale``, if ``t`` or the named parameter is not in ``list_params``,
        or if ``scale`` is not a finite, non-negative number.
    """
    if ndt_edge_shift is None:
        return
    if not isinstance(ndt_edge_shift, dict) or set(ndt_edge_shift) != {
        "param",
        "scale",
    }:
        raise ValueError(
            "`ndt_edge_shift` must be a dict with the keys 'param' and "
            f"'scale', got {ndt_edge_shift!r}."
        )
    param = ndt_edge_shift["param"]
    scale = ndt_edge_shift["scale"]
    if "t" not in list_params:
        raise ValueError(
            "`ndt_edge_shift` places the support edge relative to `t`, "
            "but `t` is not in `list_params`."
        )
    if param not in list_params:
        raise ValueError(
            f"`ndt_edge_shift` names the parameter {param!r}, "
            "which is not in `list_params`."
        )
    # bool is a subclass of int, so it is excluded explicitly.
    if (
        isinstance(scale, bool)
        or not isinstance(scale, Real)
        or not math.isfinite(scale)
        or scale < 0
    ):
        raise ValueError(
            "`ndt_edge_shift['scale']` must be a finite, non-negative "
            f"number, got {scale!r}."
        )


@dataclass
class BaseModelConfig(ABC):
    """Base configuration class for all model types."""

    # Core identification
    model_name: str
    description: str | None = None

    # Data specification
    response: list[str] | None = field(default_factory=DEFAULT_SSM_OBSERVED_DATA.copy)
    choices: tuple[int, ...] | None = None

    # Parameter specification
    list_params: list[str] | None = None
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)

    # Likelihood configuration
    loglik: LogLik | None = None
    loglik_kind: LoglikKind | None = None
    backend: Literal["jax", "pytensor"] | None = None

    # Additional data requirements
    extra_fields: list[str] | None = None

    # Random variable (simulator) for posterior predictive sampling
    rv: Any | None = None

    # Canonical per-column response metadata. Appended for positional compatibility.
    response_domains: dict[str, ResponseDomainSpec] | None = field(
        default=None, kw_only=True
    )

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

    @property
    def is_choice_only(self) -> bool:
        """Return whether the model is choice-only (no RT)."""
        return self.response is not None and len(self.response) == 1


@dataclass
class Config(BaseModelConfig):
    """Config class that stores the configurations for models."""

    rv: RandomVariable | Callable[..., np.ndarray] | None = None
    # Fields with dictionaries are automatically deepcopied
    default_priors: dict[str, ParamSpec] = field(default_factory=dict)
    # Which parameter shifts the lower edge of the response-time support, and by
    # how much (edge at t - scale * param). None keeps the edge at t.
    ndt_edge_shift: NDTEdgeShift | None = None
    # A JAX implementation of this same likelihood, f(data, *list_params), read by
    # the RLSSM registry; plain HSSM evaluates `loglik`.
    jax_loglik: Callable[..., Any] | None = None

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
                    choices = default_config.get("choices")

                    return Config(
                        model_name=model_name,
                        loglik_kind=kind,
                        response=list(default_config["response"]),
                        choices=tuple(choices) if choices is not None else None,
                        response_domains=deepcopy(
                            default_config.get("response_domains")
                        ),
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
                choices = default_config.get("choices")
                if loglik_kind in default_config["likelihoods"]:
                    loglik_config = default_config["likelihoods"][loglik_kind]
                    return Config(
                        model_name=model_name,
                        loglik_kind=loglik_kind,
                        response=list(default_config["response"]),
                        choices=tuple(choices) if choices is not None else None,
                        response_domains=deepcopy(
                            default_config.get("response_domains")
                        ),
                        list_params=default_config["list_params"],
                        description=default_config["description"],
                        **loglik_config,
                    )
                return Config(
                    model_name=model_name,
                    loglik_kind=loglik_kind,
                    response=list(default_config["response"]),
                    choices=tuple(choices) if choices is not None else None,
                    response_domains=deepcopy(default_config.get("response_domains")),
                    list_params=default_config["list_params"],
                    description=default_config["description"],
                )

            # A model HSSM does not ship but ssm-simulators knows (a built-in or
            # one registered with `register_model_config`): its ssms config
            # names the parameters, their bounds (for a LAN trained on it, the
            # training range), the choices and, where the simulator's RTs start
            # below `t`, the support edge.
            ssms_config = _get_ssms_model_config(model_name)
            if ssms_config is not None:
                return Config(
                    model_name=model_name,
                    loglik_kind=loglik_kind,
                    response=list(DEFAULT_SSM_OBSERVED_DATA),
                    list_params=list(ssms_config["params"]),
                    bounds=_bounds_from_ssms(ssms_config),
                    choices=tuple(ssms_config.get("choices", DEFAULT_SSM_CHOICES)),
                    ndt_edge_shift=ssms_config.get("ndt_edge_shift"),
                )

            return Config(
                model_name=model_name,
                loglik_kind=loglik_kind,
                response=list(DEFAULT_SSM_OBSERVED_DATA),
                choices=DEFAULT_SSM_CHOICES,
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
        if self.response_domains is not None:
            raise ValueError(
                "Provide either `response_domains` or legacy `choices`, not both."
            )

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
        if user_config.response_domains is not None:
            if user_config.choices is not None:
                raise ValueError(
                    "Provide either `response_domains` or legacy `choices`, not both."
                )
            self.response_domains = deepcopy(user_config.response_domains)
            self.choices = None
        elif user_config.choices is not None:
            if self.response_domains is not None:
                raise ValueError(
                    "Provide either `response_domains` or legacy `choices`, not both."
                )
            self.choices = user_config.choices
        if user_config.rv is not None:
            self.rv = user_config.rv

        if (
            self.loglik_kind == "approx_differentiable"
            and user_config.backend is not None
        ):
            self.backend = user_config.backend
        if user_config.ndt_edge_shift is not None:
            self.ndt_edge_shift = user_config.ndt_edge_shift

        self.default_priors |= user_config.default_priors
        self.bounds |= user_config.bounds
        self.extra_fields = user_config.extra_fields

    def validate(self) -> None:
        """Ensure that mandatory fields are not None."""
        if self.response is None:
            raise ValueError("Please provide `response` columns in the configuration.")
        self.response_domains, self.choices = _resolve_response_domains(
            self.response, self.response_domains, self.choices
        )
        if self.list_params is None:
            raise ValueError("Please provide `list_params`.")
        if self.loglik is None:
            raise ValueError("Please provide a log-likelihood function via `loglik`.")
        if self.loglik_kind == "approx_differentiable" and self.backend is None:
            raise ValueError("Please provide `backend` via `model_config`.")
        _validate_ndt_edge_shift(self.ndt_edge_shift, self.list_params)

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
        choices: int | np.integer | list[int] | tuple[int, ...] | None,
        loglik: Any = None,
    ) -> Config:
        """Build and return a validated Config for standard HSSM models.

        Resolves defaults, normalizes dict/ModelConfig overrides, applies
        choices and loglik precedence rules, then validates before returning.
        """
        # An int is the number of choices: 2 -> (-1, 1), n > 2 -> (0, ..., n - 1).
        if isinstance(choices, (int, np.integer)):
            if choices < 2:
                raise ValueError(f"`choices` must be at least 2, got {choices}.")
            choices = (-1, 1) if choices == 2 else tuple(range(choices))

        config = cls.from_defaults(model, loglik_kind)

        final_config: ModelConfig | None = None
        if model_config is not None:
            final_config = _normalize_model_config_with_choices(model_config, choices)
            config.update_config(final_config)

        # No model_config provided: apply `choices` when appropriate.
        # If caller passed a SupportedModels string, ignore explicit `choices`.
        if (
            model in get_args(SupportedModels)
            and choices is not None
            and model_config is None
        ):
            _logger.info(
                "Model string is in SupportedModels. Ignoring choices arguments."
            )

        # If model is not a supported built-in and `model_config` did not set
        # choices, prefer explicit choices or fall back to ssms-simulators lookup.
        if model not in get_args(SupportedModels) and (
            final_config is None or final_config.choices is None
        ):
            if choices is not None:
                config.update_choices(choices)
            elif (
                config.response_domains is None
                and (ssms_config := _get_ssms_model_config(model)) is not None
                and "choices" in ssms_config
            ):
                config.update_choices(tuple(ssms_config["choices"]))
                _logger.info(
                    "choices argument passed as None, "
                    "but found %s in ssms-simulators. "
                    "Using choices, from ssm-simulators configs: %s",
                    model,
                    ssms_config["choices"],
                )

        config.update_loglik(loglik)
        # A str or PathLike `loglik` is an ONNX network (local or on the
        # Hugging Face hub); "jax" is the backend every built-in LAN uses.
        if (
            config.loglik_kind == "approx_differentiable"
            and config.backend is None
            and isinstance(config.loglik, (str, PathLike))
        ):
            config.backend = "jax"
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
    rv: RandomVariable | Callable[..., np.ndarray] | None = None
    extra_fields: list[str] | None = None
    # Overrides the likelihood's own declaration when not None; see Config.
    ndt_edge_shift: NDTEdgeShift | None = None
    response_domains: dict[str, ResponseDomainSpec] | None = None


def _normalize_model_config_with_choices(
    model_config: "ModelConfig" | dict[str, Any],
    choices: list[int] | tuple[int, ...] | None,
) -> "ModelConfig":
    """Normalize a user-supplied model_config and apply choices.

    Returns a fresh `ModelConfig` instance and does not mutate the
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

    if mc.get("response_domains") is not None:
        mc["response_domains"] = deepcopy(mc["response_domains"])

    if mc.get("response_domains") is not None and (
        mc.get("choices") is not None or choices is not None
    ):
        raise ValueError(
            "Provide either `response_domains` or legacy `choices`, not both."
        )

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


def _resolve_response_domains(
    response: list[str] | tuple[str, ...] | None,
    response_domains: Mapping[str, Mapping[str, object]] | None,
    choices: list[int] | tuple[int, ...] | None,
) -> tuple[dict[str, ResponseDomainSpec], tuple[int, ...] | None]:
    """Return detached, response-ordered domain metadata and legacy choices."""
    if not response:
        raise ValueError("Please provide at least one `response` column.")
    if any(not isinstance(name, str) or not name for name in response):
        raise ValueError("Every `response` column name must be a non-empty string.")
    if len(set(response)) != len(response):
        raise ValueError("`response` column names must be unique.")

    rt_count = response.count("rt")
    if rt_count:
        if rt_count != 1 or response[0] != "rt":
            raise ValueError("RT-based models require `rt` exactly once at index zero.")
    elif len(response) != 1:
        raise ValueError(
            "Models without `rt` currently support exactly one response column."
        )

    response_columns = [name for name in response if name != "rt"]
    if not response_columns:
        raise ValueError("At least one non-RT response column is required.")

    if response_domains is None:
        if len(response_columns) != 1 or choices is None:
            raise ValueError(
                "Provide `response_domains`; legacy `choices` can describe only one "
                "non-RT response column."
            )
        raw_domains: Mapping[str, Mapping[str, object]] = {
            response_columns[0]: {"kind": "categorical", "values": choices}
        }
    else:
        if not isinstance(response_domains, Mapping):
            raise ValueError("`response_domains` must be a mapping.")
        missing = set(response_columns) - set(response_domains)
        extra = set(response_domains) - set(response_columns)
        if missing or extra:
            details = []
            if missing:
                details.append(f"missing {sorted(missing)}")
            if extra:
                details.append(f"unexpected {sorted(extra)}")
            raise ValueError(
                "`response_domains` keys must match non-RT response columns: "
                + ", ".join(details)
                + "."
            )
        raw_domains = response_domains

    resolved: dict[str, ResponseDomainSpec] = {}
    for column in response_columns:
        raw_spec = raw_domains[column]
        if not isinstance(raw_spec, Mapping):
            raise ValueError(f"Response domain for {column!r} must be a mapping.")
        kind = raw_spec.get("kind")
        if kind not in {"categorical", "continuous", "circular"}:
            raise ValueError(
                f"Response domain for {column!r} has invalid kind {kind!r}."
            )

        allowed = {"kind", "values"} if kind == "categorical" else {"kind", "bounds"}
        unknown = set(raw_spec) - allowed
        if unknown:
            raise ValueError(
                f"Response domain for {column!r} has unknown fields {sorted(unknown)}."
            )

        if kind == "categorical":
            values = raw_spec.get("values")
            if not isinstance(values, (list, tuple)) or not values:
                raise ValueError(
                    f"Categorical response domain for {column!r} requires values."
                )
            if any(
                isinstance(value, bool) or not isinstance(value, Integral)
                for value in values
            ):
                raise ValueError(f"Categorical values for {column!r} must be integers.")
            normalized_values = tuple(int(value) for value in values)
            if len(set(normalized_values)) != len(normalized_values):
                raise ValueError(f"Categorical values for {column!r} must be distinct.")
            resolved[column] = {
                "kind": "categorical",
                "values": normalized_values,
            }
            continue

        bounds = raw_spec.get("bounds")
        if "bounds" not in raw_spec and kind == "continuous":
            resolved[column] = {"kind": "continuous"}
            continue
        if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
            raise ValueError(
                f"{kind.capitalize()} response domain for {column!r} requires "
                "two bounds."
            )
        if any(
            isinstance(bound, bool) or not isinstance(bound, Real) for bound in bounds
        ):
            raise ValueError(f"Bounds for {column!r} must be real numbers.")
        lower, upper = (float(bounds[0]), float(bounds[1]))
        if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
            raise ValueError(
                f"Bounds for {column!r} must be finite and strictly increasing."
            )
        if kind == "continuous":
            resolved[column] = {"kind": "continuous", "bounds": (lower, upper)}
        else:
            resolved[column] = {"kind": "circular", "bounds": (lower, upper)}

    only_domain = next(iter(resolved.values())) if len(resolved) == 1 else None
    resolved_choices = (
        tuple(only_domain["values"])
        if only_domain is not None and only_domain["kind"] == "categorical"
        else None
    )
    if choices is not None and tuple(choices) != resolved_choices:
        raise ValueError(
            "Provide either `response_domains` or legacy `choices`, not both."
        )
    return resolved, resolved_choices
