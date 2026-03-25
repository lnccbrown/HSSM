"""RL-specific configuration classes.

This module houses `RLSSMConfig` which was previously defined in
`hssm.config`. It is intentionally lightweight and re-uses
`BaseModelConfig` from :mod:`hssm.config` to avoid duplicating core
behaviour.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .._types import LoglikKind, SupportedModels
    from ..config import ModelConfig

from ..config import BaseModelConfig

_logger = logging.getLogger("hssm")

# Local copy of required fields for RLSSM configs. Kept here so the class
# can be imported without importing the entirety of `hssm.config`'s runtime
# machinery earlier than necessary.
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


@dataclass
class RLSSMConfig(BaseModelConfig):
    """Config for reinforcement learning + sequential sampling models.

    The ``ssm_logp_func`` field holds the fully annotated JAX SSM
    log-likelihood function (an :class:`AnnotatedFunction`) that is passed
    directly to ``make_rl_logp_op``.
    """

    decision_process_loglik_kind: str = field(kw_only=True)
    learning_process_loglik_kind: str = field(kw_only=True)
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

    def validate(self) -> None:  # noqa: D102
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
