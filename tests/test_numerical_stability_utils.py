"""Tests for the utilities to ensure numerical stability of likelihood functions."""

import numpy as np
import pytest

from hssm.likelihoods.analytical import _Size, check_rt_log_domain, _log_bound_error_msg


def test_check_rt_log_domain():
    err = 1e-2
    epsilon = np.finfo(float).eps
    for size in _Size:
        bound = 1 / (np.pi * err) if size == _Size.LARGE else 1 / (8 * err**2 * np.pi)
        rt = np.array([bound * (1 + epsilon)])

        with pytest.raises(ValueError, match=r"^RTs must be less than"):
            check_rt_log_domain(rt, err, size, clip_values=False)

        _rt = check_rt_log_domain(rt, err, size)
        check_rt_log_domain(_rt, err, size, clip_values=False)


if __name__ == "__main__":
    pytest.main([__file__])
