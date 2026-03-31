"""Reinforcement-learning extensions for HSSM.

This subpackage groups components that integrate reinforcement-learning
learning rules with sequential-sampling decision models (SSMs).

Public API (import from ``hssm.rl``):

- ``RLSSM``: the RL + SSM model class implemented in :mod:`hssm.rl.rlssm`.
- ``RLSSMConfig``: the config class for RL + SSM models, implemented in
  :mod:`hssm.rl.config`.
- ``validate_balanced_panel``: panel-balance utility in :mod:`hssm.rl.utils`.

RL likelihood builders live in :mod:`hssm.rl.likelihoods.builder` and include
helpers such as :func:`~hssm.rl.likelihoods.builder.make_rl_logp_func` and
:func:`~hssm.rl.likelihoods.builder.make_rl_logp_op`.

"""

from .config import RLSSMConfig
from .rlssm import RLSSM
from .utils import validate_balanced_panel

__all__ = [
    "RLSSM",
    "RLSSMConfig",
    "validate_balanced_panel",
]
