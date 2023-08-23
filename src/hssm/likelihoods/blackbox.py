"""Black box likelihoods written in Cython for "ddm" and "ddm_sdv" models."""

from __future__ import annotations

import hddm_wfpt
import numpy as np


def logp_ddm_bbox(data: np.ndarray, v, a, z, t) -> np.ndarray:
    """Compute blackbox log-likelihoods for ddm models."""
    x = (data[:, 0] * data[:, 1]).astype(np.float64)
    size = len(data)

    v, a, z, t = [_broadcast(param, size) for param in [v, a, z, t]]
    zeros = np.zeros(size, dtype=np.float64)

    return hddm_wfpt.wfpt.wiener_logp_array(
        x=x,
        v=v,
        sv=zeros,
        a=a * 2,  # Ensure compatibility with HSSM.
        z=z,
        sz=zeros,
        t=t,
        st=zeros,
        err=1e-8,
    ).astype(data.dtype)


def logp_ddm_sdv_bbox(data: np.ndarray, v, a, z, t, sv) -> np.ndarray:
    """Compute blackbox log-likelihoods for ddm models."""
    x = (data[:, 0] * data[:, 1]).astype(np.float64)
    size = len(x)

    v, a, z, t, sv = [_broadcast(param, size) for param in [v, a, z, t, sv]]
    zeros = np.zeros(size, dtype=np.float64)

    return hddm_wfpt.wfpt.wiener_logp_array(
        x=x,
        v=v,
        sv=zeros,
        a=a * 2,  # Ensure compatibility with HSSM.
        z=z,
        sz=zeros,
        t=t,
        st=zeros,
        err=1e-8,
    ).astype(data.dtype)


def _broadcast(x: float | np.ndarray, size: int):
    """Broadcast a scalar or an array to size of `size`."""
    return np.broadcast_to(np.array(x, dtype=np.float64), size)
