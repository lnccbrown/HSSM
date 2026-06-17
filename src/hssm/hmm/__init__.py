"""Regime-switching SSM extensions for HSSM.

This subpackage groups the components that implement :class:`RSSSM`, a
regime-switching sequential sampling model where the per-trial SSM emission is
modulated by a hidden Markov chain over cognitive regimes.

Public API (import from ``hssm.hmm`` or, for the class, from ``hssm``):

- ``RSSSM``: the regime-switching SSM class (:mod:`hssm.hmm.rsssm`).
- ``RSSSMConfig``: its config class (:mod:`hssm.hmm.config`).
- The spec dataclasses needed to build an ``RSSSMConfig`` by hand
  (:mod:`hssm.hmm.specs`).
"""

from .config import RSSSMConfig
from .rsssm import RSSSM
from .specs import (
    AutoOrdering,
    DirichletConcentration,
    DirichletInitialDistribution,
    FixedInitialDistribution,
    FullPooling,
    NoOrdering,
    NoPooling,
    OrderByParam,
    StickyDirichlet,
    UniformInitialDistribution,
)

__all__ = [
    "RSSSM",
    "RSSSMConfig",
    "StickyDirichlet",
    "DirichletConcentration",
    "UniformInitialDistribution",
    "FixedInitialDistribution",
    "DirichletInitialDistribution",
    "AutoOrdering",
    "OrderByParam",
    "NoOrdering",
    "FullPooling",
    "NoPooling",
]
