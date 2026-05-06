"""Reinforcement-learning extensions for HSSM.

This subpackage groups components that integrate reinforcement-learning
learning rules with sequential-sampling decision models (SSMs).

Public API (import from ``hssm.rl``):

- ``RLSSM``: the public RL + SSM model class in :mod:`hssm.rl.rlssm`.
- ``_RLSSM``: the internal base class that requires a fully built config.
- ``RLSSMConfig``: the config class for RL + SSM models in :mod:`hssm.rl.config`.
- ``get_rlssm_model_config``: factory that builds a config from a named model.
- ``register_rlssm_model``: register a custom named RLSSM model.
- ``register_ssm``: register a custom SSM base logp function.
- ``validate_balanced_panel``: panel-balance utility in :mod:`hssm.rl.utils`.

RL likelihood builders live in :mod:`hssm.rl.likelihoods.builder` and include
helpers such as :func:`~hssm.rl.likelihoods.builder.make_rl_logp_func` and
:func:`~hssm.rl.likelihoods.builder.make_rl_logp_op`.

"""

from .config import RLSSMConfig
from .registry import get_rlssm_model_config, register_rlssm_model, register_ssm
from .rlssm import _RLSSM, RLSSM
from .utils import validate_balanced_panel

__all__ = [
    "RLSSM",
    "_RLSSM",
    "RLSSMConfig",
    "get_rlssm_model_config",
    "register_rlssm_model",
    "register_ssm",
    "validate_balanced_panel",
]
