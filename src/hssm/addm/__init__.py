"""Attentional Drift Diffusion Model (aDDM) integration for HSSM.

Exposes ``aDDM`` and ``aDDMConfig`` (peers of ``RLSSM`` / ``RLSSMConfig``) plus
the attention-process registry. The vendored JAX likelihood kernel lives under
``likelihoods/jax/``.
"""

from .addm import aDDM
from .attention_process import (
    ATTENTION_PROCESSES,
    resolve_attention_process,
    standard_alternating,
)
from .config import aDDMConfig

__all__ = [
    "aDDM",
    "aDDMConfig",
    "ATTENTION_PROCESSES",
    "resolve_attention_process",
    "standard_alternating",
]
