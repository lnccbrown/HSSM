"""RL-specific configuration classes.

This module houses `RLSSMConfig` which was previously defined in
`hssm.config`. It is intentionally lightweight and re-uses
`BaseModelConfig` from :mod:`hssm.config` to avoid duplicating core
behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._types import LoglikKind, SupportedModels
    from ..config import ModelConfig

from ..config import BaseModelConfig

_logger = logging.getLogger("hssm")


@dataclass
class RLSSMConfig(BaseModelConfig):
    """Config for reinforcement learning + sequential sampling models.

    The ``ssm_logp_func`` field holds the fully annotated JAX SSM
    log-likelihood function (an :class:`AnnotatedFunction`) that is passed
    directly to ``make_rl_logp_op``.
    """

    decision_process_loglik_kind: str = field(kw_only=True)
    learning_process_kind: str = field(kw_only=True)
    params_default: list[float] = field(kw_only=True)
    decision_process: str | "ModelConfig" = field(kw_only=True)
    learning_process: dict[str, Any] = field(kw_only=True)
    ssm_logp_func: Any = field(default=None, kw_only=True)

    def __post_init__(self):  # noqa: D105
        if self.loglik_kind is None:
            self.loglik_kind = "approx_differentiable"
            _logger.debug(
                "RLSSMConfig: loglik_kind not specified; "
                "defaulting to 'approx_differentiable'."
            )

    @classmethod
    def from_defaults(  # noqa: D102
        cls, model_name: "SupportedModels" | str, loglik_kind: "LoglikKind" | None
    ) -> "RLSSMConfig":
        raise NotImplementedError(
            "RLSSMConfig does not support from_defaults(). "
            "Use RLSSMConfig.from_rlssm_dict() or the constructor directly."
        )

    @classmethod
    def from_rlssm_dict(cls, config_dict: dict[str, Any]) -> "RLSSMConfig":  # noqa: D102
        # field exceptions to allow the constructor to fill in defaults
        field_exceptions = ("loglik", "loglik_kind", "backend")
        required_fields = [
            f.name for f in fields(cls) if f.name not in field_exceptions
        ]
        for field_name in required_fields:
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
            learning_process_kind=config_dict["learning_process_kind"],
        )

    def validate(self) -> None:  # noqa: D102
        if self.response is None:
            raise ValueError("Please provide `response` columns in the configuration.")
        if self.list_params is None:
            raise ValueError("Please provide `list_params` in the configuration.")
        if self.choices is None:
            raise ValueError("Please provide `choices` in the configuration.")
        if self.decision_process is None:
            raise ValueError("Please specify a `decision_process`.")

        logpfunc = self.ssm_logp_func
        if logpfunc is None:
            raise ValueError(
                "Please provide `ssm_logp_func`: the fully annotated JAX SSM "
                "log-likelihood function required by `make_rl_logp_op`."
            )
        if not callable(logpfunc):
            raise ValueError(
                f"`ssm_logp_func` must be a callable, but got {type(logpfunc)!r}."
            )
        missing_attrs = [
            attr
            for attr in ("inputs", "outputs", "computed")
            if not hasattr(logpfunc, attr)
        ]
        if missing_attrs:
            raise ValueError(
                "`ssm_logp_func` must be decorated with `@annotate_function` "
                "so that it carries the attributes required by `make_rl_logp_op`. "
                f"Missing attribute(s): {missing_attrs}. "
            )

        if not isinstance(logpfunc.computed, dict) or not all(
            callable(v) for v in logpfunc.computed.values()
        ):
            raise ValueError(
                "`ssm_logp_func.computed` must be a dictionary with callable values."
            )

        if self.params_default and self.list_params:
            if len(self.params_default) != len(self.list_params):
                raise ValueError(
                    f"params_default length ({len(self.params_default)}) doesn't "
                    f"match list_params length ({len(self.list_params)})"
                )

        if self.list_params:
            missing_bounds = [p for p in self.list_params if p not in self.bounds]
            if missing_bounds:
                raise ValueError(
                    f"Missing bounds for parameter(s): {missing_bounds}. "
                    "Every parameter in `list_params` must have a corresponding "
                    "entry in `bounds`."
                )

    def get_defaults(  # noqa: D102
        self, param: str
    ) -> tuple[float | None, tuple[float, float] | None]:
        return None, self.bounds.get(param)
