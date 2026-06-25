"""Configuration for the attentional Drift Diffusion Model (aDDM).

``aDDMConfig`` mirrors :class:`hssm.rl.config.RLSSMConfig` on
:class:`hssm.config.BaseModelConfig`. Like ``RLSSMConfig``, aDDM is **not** a
``SupportedModels`` entry — this config is constructed directly (e.g.
``hssm.aDDM(data=..., model_config=aDDMConfig(...))`` in Commit 4) and never
resolved through ``get_default_model_config``.

Every field carries a default, so ``aDDMConfig()`` yields a complete default
configuration (the factory the subclass uses). ``loglik`` and ``backend`` are
inherited as ``None`` and injected by ``aDDM.__init__`` via
``dataclasses.replace`` in Commit 4.
"""

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from typing import Any

from ..config import BaseModelConfig
from .attention_process import resolve_attention_process


@dataclass
class aDDMConfig(BaseModelConfig):
    """Config for the attentional DDM (subclass formulation).

    Parameters (sampled) are ``[eta, kappa, a, b, x0]``; the per-trial covariates
    ``[r1, r2, flag, sacc_array, d, sigma]`` are supplied as ``extra_fields`` (not
    sampled), ordered to match the kernel's positional covariate slots.
    """

    model_name: str = "addm"
    description: str | None = "Attentional Drift Diffusion Model"
    response: list[str] = field(default_factory=lambda: ["rt", "response"])
    choices: tuple[int, ...] = (-1, 1)
    list_params: list[str] = field(
        default_factory=lambda: ["eta", "kappa", "a", "b", "x0"]
    )
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "eta": (0.0, 1.0),
            "kappa": (0.0, 5.0),
            "a": (0.1, 3.0),
            "b": (0.0, 3.0),
            "x0": (-1.0, 1.0),
        }
    )
    extra_fields: list[str] | None = field(
        default_factory=lambda: ["r1", "r2", "flag", "sacc_array", "d", "sigma"]
    )
    loglik_kind: str = "approx_differentiable"
    params_default: list[float] = field(
        default_factory=lambda: [0.3, 1.0, 1.0, 2.0, 0.0]
    )
    attention_process: str | Callable = "standard_alternating"
    # ``loglik`` and ``backend`` are inherited (default ``None``) and injected by
    # ``aDDM.__init__`` via ``dataclasses.replace`` in Commit 4 — not redeclared here.

    def validate(self) -> None:
        """Validate the configuration (mirrors ``RLSSMConfig.validate``)."""
        if not self.list_params:
            raise ValueError("Please provide `list_params` in the configuration.")
        # Raises ValueError (unknown name) / TypeError (bad type) on failure.
        resolve_attention_process(self.attention_process)
        if self.params_default and len(self.params_default) != len(self.list_params):
            raise ValueError(
                f"params_default length ({len(self.params_default)}) doesn't match "
                f"list_params length ({len(self.list_params)})."
            )
        missing = [p for p in self.list_params if p not in self.bounds]
        if missing:
            raise ValueError(
                f"Missing bounds for parameter(s): {missing}. Every parameter in "
                "`list_params` must have a corresponding entry in `bounds`."
            )
        if not self.extra_fields:
            raise ValueError(
                "aDDM requires `extra_fields` (the per-trial covariates "
                "r1, r2, flag, sacc_array, d, sigma)."
            )

    def get_defaults(self, param: str) -> tuple[None, tuple[float, float] | None]:
        """Return ``(prior, bounds)`` for ``param`` (no prior; bounds from config).

        Implements the abstract ``BaseModelConfig.get_defaults`` with the same
        contract as ``RLSSMConfig``: ``params_default`` holds initialisation
        values, not priors, so the prior slot is ``None``.
        """
        return None, self.bounds.get(param)

    @classmethod
    def from_addm_dict(cls, config_dict: dict[str, Any]) -> "aDDMConfig":
        """Build an ``aDDMConfig`` from a dict, ignoring unknown keys."""
        field_names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in config_dict.items() if k in field_names})
