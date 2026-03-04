"""Reinforcement learning extensions for HSSM.

This sub-package provides:

- :class:`~hssm.rl.rlssm.RLSSM` — the RL + SSM model class.
- :func:`~hssm.rl.utils.validate_balanced_panel` — panel-balance utility.
- :mod:`hssm.rl.likelihoods` — log-likelihood builders
  (:func:`~hssm.rl.likelihoods.builder.make_rl_logp_func`,
  :func:`~hssm.rl.likelihoods.builder.make_rl_logp_op`).
"""

from .rlssm import RLSSM
from .utils import validate_balanced_panel

__all__ = [
    "RLSSM",
    "validate_balanced_panel",
]
