"""Likelihood components for regime-switching SSMs.

- :mod:`hssm.hmm.likelihoods.forward` — L1: the batched forward recursion.
- :mod:`hssm.hmm.likelihoods.emissions` — L2: per-regime SSM emission logp.
- :mod:`hssm.hmm.likelihoods.builder` — L3: composes L1 + L2 into the
  forward-marginal ``pm.Potential``.
"""

from .builder import make_hmm_logp_op
from .forward import forward_log_marginal

__all__ = ["make_hmm_logp_op", "forward_log_marginal"]
