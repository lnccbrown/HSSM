"""Vendored pure-JAX aDDM log-likelihood kernel.

Frozen in-tree copy of ``efficient_fpt.jax`` (see ``NOTICE.md`` for provenance and
the HSSM-authored modifications). This module exposes a deliberately narrow
surface; nothing outside ``hssm.addm`` should import from it directly, so a
future re-vendor stays a single-directory change.
"""

from .addm_helpers import _build_addm_mu_array_data
from .batch import (
    compute_addm_loglikelihoods,  # primary batched kernel (alias of ..._batchscan)
    compute_addm_loglikelihoods_from_mu,  # core wrapper for custom attention processes
    make_addm_nll_function,  # parity/reference only
)
from .multi_stage import compute_addm_logfptd  # parity tests only
from .utils import get_jax_dtype, set_jax_precision

__all__ = [
    "compute_addm_loglikelihoods",
    "compute_addm_loglikelihoods_from_mu",
    "make_addm_nll_function",
    "compute_addm_logfptd",
    "_build_addm_mu_array_data",
    "set_jax_precision",
    "get_jax_dtype",
]
