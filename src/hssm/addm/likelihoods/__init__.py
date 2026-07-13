"""aDDM likelihood machinery.

Holds the vendored pure-JAX kernel under ``jax/`` and the likelihood builder
(:func:`make_addm_logp_func` / :func:`make_addm_logp_op`) that wraps it as a
differentiable PyTensor ``Op``.
"""

from .builder import make_addm_logp_func, make_addm_logp_op

__all__ = [
    "make_addm_logp_func",
    "make_addm_logp_op",
]
