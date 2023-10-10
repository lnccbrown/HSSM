"""Black box likelihoods written in Cython for "ddm" and "ddm_sdv" models."""

import numpy as np
from hddm_wfpt import wfpt


def hddm_to_hssm(func):
    """Make HDDM likelihood function compatible with HSSM."""

    def outer(data: np.ndarray, *args, **kwargs):
        x = data[:, 0] * np.where(data[:, 1] == 1, 1.0, -1.0).astype(np.float64)
        size = len(data)

        args_list = [_broadcast(param, size) for param in args]
        kwargs = {k: _broadcast(param, size) for k, param in kwargs.items()}

        logp = func(x, *args_list, *kwargs)
        logp = np.where(np.isfinite(logp), logp, -66.1)

        return logp.astype(data.dtype)

    return outer


@hddm_to_hssm
def logp_ddm_bbox(data: np.ndarray, v, a, z, t) -> np.ndarray:
    """Compute blackbox log-likelihoods for ddm models."""
    size = len(data)
    zeros = np.zeros(size, dtype=np.float64)

    return wfpt.wiener_logp_array(
        x=data,
        v=v,
        sv=zeros,
        a=a * 2,  # Ensure compatibility with HSSM.
        z=z,
        sz=zeros,
        t=t,
        st=zeros,
        err=1e-8,
    )


@hddm_to_hssm
def logp_ddm_sdv_bbox(data: np.ndarray, v, a, z, t, sv) -> np.ndarray:
    """Compute blackbox log-likelihoods for ddm_sdv models."""
    size = len(data)
    zeros = np.zeros(size, dtype=np.float64)

    return wfpt.wiener_logp_array(
        x=data,
        v=v,
        sv=sv,
        a=a * 2,  # Ensure compatibility with HSSM.
        z=z,
        sz=zeros,
        t=t,
        st=zeros,
        err=1e-8,
    )


@hddm_to_hssm
def logp_full_ddm(data: np.ndarray, v, a, z, t, sv, sz, st):
    """Compute blackbox log-likelihoods for full_ddm models."""
    return wfpt.wiener_logp_array(
        x=data,
        v=v,
        sv=sv,
        a=a * 2,  # Ensure compatibility with HSSM.
        z=z,
        sz=sz,
        t=t,
        st=st,
        err=1e-8,
    )


def _broadcast(x: float | np.ndarray, size: int):
    """Broadcast a scalar or an array to size of `size`."""
    return np.broadcast_to(np.array(x, dtype=np.float64), size)
