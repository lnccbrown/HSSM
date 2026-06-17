"""Typed-union spec dataclasses for :class:`~hssm.hmm.config.RSSSMConfig`.

Each user-facing config field on ``RSSSMConfig`` (transition prior, initial
distribution, ordering, pooling) is a *typed union of small dataclasses*.  Each
union ships one or two concrete cases in v1 and leaves room for the deferred
v2/v3 variants without ever changing the constructor signature (see the design
doc, §4.4 and §8).

These dataclasses are the **internal** representation.  Users may pass them
directly (advanced path) or use the HSSM-style prior dict / value shorthand on
the main constructor; :func:`resolve_transition_prior` and
:func:`resolve_initial_distribution` normalise the shorthand into these objects
(mirroring HSSM's ``UserParam -> Param`` pipeline).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, Union

import bambi as bmb
import numpy as np

_logger = logging.getLogger("hssm")

# ---------------------------------------------------------------------------
# Transition-matrix prior
# ---------------------------------------------------------------------------


@dataclass
class StickyDirichlet:
    """Sticky-diagonal Dirichlet prior on each row of ``P``.

    The concentration matrix has ``diag`` on the diagonal and ``offdiag``
    everywhere else, encoding the belief that regimes persist (switches are
    rare).  This is the v1 default.
    """

    diag: float = 20.0
    offdiag: float = 2.0

    def concentration(self, K: int) -> np.ndarray:
        """Return the ``(K, K)`` Dirichlet concentration matrix."""
        alpha = np.full((K, K), float(self.offdiag))
        np.fill_diagonal(alpha, float(self.diag))
        return alpha


@dataclass
class DirichletConcentration:
    """Explicit Dirichlet concentration for the transition matrix.

    ``alpha`` may be a full ``(K, K)`` matrix (one concentration vector per
    row), a single ``(K,)`` vector (shared across rows), or a scalar
    (broadcast to every entry).
    """

    alpha: Any

    def concentration(self, K: int) -> np.ndarray:
        """Return the ``(K, K)`` Dirichlet concentration matrix."""
        alpha = np.asarray(self.alpha, dtype=float)
        if alpha.ndim == 0:
            return np.full((K, K), float(alpha))
        if alpha.ndim == 1:
            if alpha.shape[0] != K:
                raise ValueError(
                    f"transition_prior alpha vector has length {alpha.shape[0]}, "
                    f"expected K={K}."
                )
            return np.tile(alpha, (K, 1))
        if alpha.shape != (K, K):
            raise ValueError(
                f"transition_prior alpha matrix has shape {alpha.shape}, "
                f"expected ({K}, {K})."
            )
        return alpha


# CovariateDrivenTransition is the documented v2 hook; not shipped in v1.
TransitionPriorSpec = Union[StickyDirichlet, DirichletConcentration]


# ---------------------------------------------------------------------------
# Initial-state distribution
# ---------------------------------------------------------------------------


@dataclass
class UniformInitialDistribution:
    """Fixed uniform initial-state distribution ``pi0 = 1/K`` (v1 default)."""

    def pi0_value(self, K: int) -> np.ndarray:
        """Return the fixed ``(K,)`` initial distribution."""
        return np.full(K, 1.0 / K)


@dataclass
class FixedInitialDistribution:
    """Fixed (non-estimable) initial-state distribution from a user vector."""

    pi0: Any

    def pi0_value(self, K: int) -> np.ndarray:
        """Return the fixed ``(K,)`` initial distribution."""
        pi0 = np.asarray(self.pi0, dtype=float)
        if pi0.shape != (K,):
            raise ValueError(
                f"initial_distribution pi0 has shape {pi0.shape}, expected ({K},)."
            )
        if not np.isclose(pi0.sum(), 1.0):
            raise ValueError(
                f"initial_distribution pi0 must sum to 1, got {pi0.sum()}."
            )
        return pi0


@dataclass
class DirichletInitialDistribution:
    """Estimable (global) initial-state distribution ``pi0 ~ Dirichlet(alpha)``.

    ``pi0`` is informed only by each participant's first trial, so it is weakly
    identified at small ``N`` (it recovers the empirical first-state frequency).
    It is global across participants in v1.
    """

    alpha: Any = None

    def concentration(self, K: int) -> np.ndarray:
        """Return the ``(K,)`` Dirichlet concentration vector."""
        if self.alpha is None:
            return np.ones(K)
        alpha = np.asarray(self.alpha, dtype=float)
        if alpha.ndim == 0:
            return np.full(K, float(alpha))
        if alpha.shape != (K,):
            raise ValueError(
                f"initial_distribution alpha has shape {alpha.shape}, expected ({K},)."
            )
        return alpha


InitialDistributionSpec = Union[
    UniformInitialDistribution,
    FixedInitialDistribution,
    DirichletInitialDistribution,
]


# ---------------------------------------------------------------------------
# Label-switching / ordering
# ---------------------------------------------------------------------------


@dataclass
class AutoOrdering:
    """Automatically pick an anchor switching parameter and order it (default).

    The anchor's length-K prior is declared with PyMC's ``ordered`` transform,
    making permuted-label modes unreachable (ascending: regime 0 is the
    lowest-anchor-value regime).
    """


@dataclass
class OrderByParam:
    """Order on a user-chosen anchor parameter and direction."""

    name: str
    direction: Literal["asc", "desc"] = "asc"


@dataclass
class NoOrdering:
    """Escape hatch: no ordering transform (posteriors may be multi-modal)."""


OrderingSpec = Union[AutoOrdering, OrderByParam, NoOrdering]


# ---------------------------------------------------------------------------
# Pooling across participants
# ---------------------------------------------------------------------------


@dataclass
class FullPooling:
    """Global SSM parameters shared across participants (v1 default)."""


@dataclass
class NoPooling:
    """Independent per-participant SSM parameters (``pooling="none"``).

    ``P`` and ``pi0`` remain global.  Partial (hierarchical) pooling is the
    deferred Phase 6.1 extension.
    """


PoolingSpec = Union[FullPooling, NoPooling]


# ---------------------------------------------------------------------------
# Resolvers: HSSM-style shorthand -> internal spec dataclasses
# ---------------------------------------------------------------------------


def _prior_dict_alpha(spec: dict[str, Any] | bmb.Prior) -> Any:
    """Extract the Dirichlet ``alpha`` from a prior dict / ``bmb.Prior``."""
    if isinstance(spec, bmb.Prior):
        name, args = spec.name, spec.args
    else:
        spec = dict(spec)
        name = spec.pop("name", None)
        args = spec
    if name is not None and name != "Dirichlet":
        raise ValueError(
            f"Only the 'Dirichlet' distribution is supported for simplex/matrix "
            f"priors on P / pi0, got {name!r}."
        )
    if "alpha" not in args:
        raise ValueError("Dirichlet prior spec must provide an 'alpha' concentration.")
    return args["alpha"]


def resolve_transition_prior(
    spec: TransitionPriorSpec | dict[str, Any] | bmb.Prior | None,
) -> TransitionPriorSpec:
    """Normalise a user transition-prior input into a spec dataclass.

    Accepts ``None`` (sticky-diagonal default), a ``StickyDirichlet`` /
    ``DirichletConcentration`` instance, a sticky-shorthand dict
    (``{"sticky_diag": ..., "sticky_offdiag": ...}``), or an HSSM-style
    ``{"name": "Dirichlet", "alpha": ...}`` prior dict / ``bmb.Prior``.
    """
    if spec is None:
        return StickyDirichlet()
    if isinstance(spec, (StickyDirichlet, DirichletConcentration)):
        return spec
    if isinstance(spec, bmb.Prior):
        return DirichletConcentration(alpha=_prior_dict_alpha(spec))
    if isinstance(spec, dict):
        if "sticky_diag" in spec or "sticky_offdiag" in spec:
            return StickyDirichlet(
                diag=spec.get("sticky_diag", 20.0),
                offdiag=spec.get("sticky_offdiag", 2.0),
            )
        return DirichletConcentration(alpha=_prior_dict_alpha(spec))
    raise TypeError(
        f"Unsupported transition_prior input of type {type(spec)!r}. Provide a "
        "spec dataclass, a sticky-shorthand dict, or a Dirichlet prior dict."
    )


def resolve_initial_distribution(
    spec: InitialDistributionSpec | dict[str, Any] | bmb.Prior | str | Any | None,
) -> InitialDistributionSpec:
    """Normalise a user initial-distribution input into a spec dataclass.

    Accepts ``None`` / ``"uniform"`` (uniform default), a spec dataclass, an
    HSSM-style ``{"name": "Dirichlet", "alpha": ...}`` dict / ``bmb.Prior``
    (estimable), or a fixed vector (list / ``np.ndarray``).
    """
    if spec is None or (isinstance(spec, str) and spec == "uniform"):
        return UniformInitialDistribution()
    if isinstance(
        spec,
        (
            UniformInitialDistribution,
            FixedInitialDistribution,
            DirichletInitialDistribution,
        ),
    ):
        return spec
    if isinstance(spec, bmb.Prior):
        return DirichletInitialDistribution(alpha=_prior_dict_alpha(spec))
    if isinstance(spec, dict):
        return DirichletInitialDistribution(alpha=_prior_dict_alpha(spec))
    if isinstance(spec, (list, tuple, np.ndarray)):
        return FixedInitialDistribution(pi0=spec)
    raise TypeError(
        f"Unsupported initial_distribution input of type {type(spec)!r}. Provide "
        '"uniform", a fixed vector, a Dirichlet prior dict, or a spec dataclass.'
    )


def resolve_ordering(
    spec: OrderingSpec | str | dict[str, Any] | None,
) -> OrderingSpec:
    """Normalise a user ordering input into a spec dataclass."""
    if spec is None:
        return AutoOrdering()
    if isinstance(spec, (AutoOrdering, OrderByParam, NoOrdering)):
        return spec
    if isinstance(spec, str):
        if spec == "auto":
            return AutoOrdering()
        if spec == "none":
            return NoOrdering()
        return OrderByParam(name=spec)
    if isinstance(spec, dict):
        return OrderByParam(**spec)
    raise TypeError(f"Unsupported ordering input of type {type(spec)!r}.")


def resolve_pooling(spec: PoolingSpec | str | None) -> PoolingSpec:
    """Normalise a user pooling input into a spec dataclass."""
    if spec is None:
        return FullPooling()
    if isinstance(spec, (FullPooling, NoPooling)):
        return spec
    if isinstance(spec, str):
        if spec in ("full", "none"):
            return FullPooling() if spec == "full" else NoPooling()
        raise ValueError(f"Unknown pooling option {spec!r}; use 'full' or 'none'.")
    raise TypeError(f"Unsupported pooling input of type {type(spec)!r}.")
