"""Configuration class for regime-switching SSMs (:class:`RSSSM`).

``RSSSMConfig`` extends :class:`~hssm.config.BaseModelConfig` with the fields
that define the Markov-chain structure (``K``, ``switching_params``, the
transition prior, the initial distribution), the emission (``model`` /
``loglik_kind`` / ``emission_logp_func``), the label-switching ``ordering``,
and the cross-participant ``pooling``.  See the design doc §5.1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from ..config import BaseModelConfig
from .specs import (
    AutoOrdering,
    FullPooling,
    InitialDistributionSpec,
    OrderingSpec,
    PoolingSpec,
    TransitionPriorSpec,
    UniformInitialDistribution,
)

if TYPE_CHECKING:
    from .._types import LoglikKind

_logger = logging.getLogger("hssm")


@dataclass
class RSSSMConfig(BaseModelConfig):
    """Config for regime-switching sequential sampling models.

    Parameters
    ----------
    K
        Number of hidden regimes (``>= 2``).
    switching_params
        SSM parameters that are *inferred per regime* (each becomes a length-K
        vector).  Parameters not listed here are shared (scalar) unless a
        per-regime fixed value is supplied via ``param_specs``.
    model
        The SSM identifier (e.g. ``"ddm"``) or a pre-built ``BaseModelConfig``.
    transition_prior
        The (resolved) transition-matrix prior spec.
    initial_distribution
        The (resolved) initial-state distribution spec.  Defaults to uniform.
    ordering
        The (resolved) label-switching spec.  Defaults to ``AutoOrdering``.
    pooling
        The (resolved) cross-participant pooling spec.  Defaults to
        ``FullPooling``.
    emission_logp_func
        The resolved per-regime emission log-density callable.  When ``None``
        it is resolved from ``model`` + ``loglik_kind`` at build time.
    param_specs
        Per-parameter user inputs implementing the three-mode rule: a scalar
        means *shared*, a length-K list means *fixed per regime*, a prior dict
        / ``bmb.Prior`` means *inferred* with that prior.  Parameters absent
        here fall back to the SSM model's default / a bounds-based prior.
    """

    # --- Markov chain structure ---
    K: int = field(default=2, kw_only=True)
    switching_params: list[str] = field(default_factory=list, kw_only=True)
    transition_prior: TransitionPriorSpec | None = field(default=None, kw_only=True)
    initial_distribution: InitialDistributionSpec = field(
        default_factory=UniformInitialDistribution, kw_only=True
    )

    # --- Emission ---
    model: str | BaseModelConfig | None = field(default=None, kw_only=True)
    emission_logp_func: Any = field(default=None, kw_only=True)
    # loglik_kind / backend are inherited from BaseModelConfig.

    # --- Label-switching ---
    ordering: OrderingSpec = field(default_factory=AutoOrdering, kw_only=True)

    # --- Pooling across participants ---
    pooling: PoolingSpec = field(default_factory=FullPooling, kw_only=True)

    # --- Per-parameter input specs (three-mode rule) ---
    param_specs: dict[str, Any] = field(default_factory=dict, kw_only=True)

    @classmethod
    def from_defaults(  # noqa: D102
        cls, model_name: str, loglik_kind: "LoglikKind" | None
    ) -> "RSSSMConfig":
        raise NotImplementedError(
            "RSSSMConfig is constructed by RSSSM.__init__ or directly; it does "
            "not support from_defaults()."
        )

    def get_defaults(  # noqa: D102
        self, param: str
    ) -> tuple[None, tuple[float, float] | None]:
        return None, self.bounds.get(param)

    def validate(self) -> None:
        """Validate the regime-switching configuration."""
        if self.K < 2:
            raise ValueError(f"K must be >= 2, got K={self.K}.")
        if self.list_params is None:
            raise ValueError(
                "list_params must be populated from the SSM model before validation."
            )
        unknown = [p for p in self.switching_params if p not in self.list_params]
        if unknown:
            raise ValueError(
                f"switching_params {unknown} are not parameters of model "
                f"{self.model!r} (list_params={self.list_params})."
            )
        # The emission must be set or resolvable from the SSM model.
        if self.emission_logp_func is None and self.model is None:
            raise ValueError(
                "Either `model` (to resolve the emission) or a pre-built "
                "`emission_logp_func` must be provided."
            )
        # The transition prior must be shape-consistent with K (eagerly, so a
        # mismatched concentration matrix is caught here, not at build time).
        if self.transition_prior is not None:
            self.transition_prior.concentration(self.K)
        # Fixed-per-regime values supplied through param_specs must have length K.
        for name, spec in self.param_specs.items():
            if name not in self.list_params:
                raise ValueError(
                    f"param_specs key {name!r} is not a parameter of the SSM "
                    f"(list_params={self.list_params})."
                )
            # Fixed-per-regime values arrive as a list/tuple or a 1-D ndarray; a
            # scalar (shared) and a 0-D ndarray are handled elsewhere.
            is_fixed_vector = isinstance(spec, (list, tuple)) or (
                isinstance(spec, np.ndarray) and spec.ndim == 1
            )
            if is_fixed_vector and len(spec) != self.K:
                raise ValueError(
                    f"Fixed-per-regime value for {name!r} has length {len(spec)}, "
                    f"expected K={self.K}."
                )
