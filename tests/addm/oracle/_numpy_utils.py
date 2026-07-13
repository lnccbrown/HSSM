# efpt NumPy reference backend -- vendored TEST ORACLE (not shipped in the package).
# Source: efficient-fpt @ d97a451479141acef845195610f0f9d85824844e
#         MIT License, Copyright (c) 2025 Sicheng Liu.
# Vendored verbatim except import retargeting to flatten the backend into this
# self-contained  package. Kept at efpt's ORIGINAL DEFAULT_TRUNC_NUM=100
# so it is an INDEPENDENT numerical reference for the HSSM-vendored jax kernel
# (do not sync to that kernel's local modifications).

"""NumPy utility helpers for efficient-fpt."""

from __future__ import annotations

import numpy as np


def positive_log(values):
    """Return log(x) for x > 0, -inf for x <= 0, and nan when x is nan."""
    values = np.asarray(values, dtype=np.float64)
    result = np.where(values > 0.0, np.log(np.maximum(values, 1e-300)), -np.inf)
    result = np.where(np.isnan(values), np.nan, result)
    return float(result) if result.ndim == 0 else result
