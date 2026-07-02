import sys

import hssm
import pytest
import numpy as np

hssm.set_floatX("float32")

PARAMETER_NAMES = "draws,safe_mode,inplace"
PARAMETER_GRID = [
    (1, False, False),
    (1, True, False),
    # (None, False, False), # slow test...
    (None, True, False),
    # (50, False, False),
    (50, True, False),
    # (np.arange(500), False, False), # very slow to test
    (np.arange(500), True, False),
    (1, False, True),
    (1, True, True),
    # (None, False, True), # very slow to test
    (None, True, True),
    (50, False, True),
    (50, True, True),
    (np.arange(50), False, True),
    (np.arange(500), True, True),
    ([1, 2, 3, 4, 5], False, False),
    ([1, 2, 3, 4, 5], True, False),
    ([1, 2, 3, 4, 5], False, True),
    ([1, 2, 3, 4, 5], True, True),
]


@pytest.mark.xfail(
    sys.version_info >= (3, 14),
    reason="sample_posterior_predictive fails on 3.14 with cpickle issue",
    strict=True,  # This will let us know in the future when this is fixed
)
@pytest.mark.slow
@pytest.mark.parametrize(PARAMETER_NAMES, PARAMETER_GRID)
def test_sample_posterior_predictive(cav_dt, cavanagh_test, draws, safe_mode, inplace):
    """Test sample_posterior_predictive method."""

    model = hssm.HSSM(
        data=cavanagh_test,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                },
                "formula": "v ~ theta + (1|participant_id)",
                "link": "identity",
            },
        ],
    )  # Doesn't matter what model or data we use here
    if "posterior_predictive" in cav_dt:
        del cav_dt["posterior_predictive"]
    cav_dt_copy = cav_dt.copy()

    posterior_predictive = model.sample_posterior_predictive(
        dt=cav_dt_copy, draws=draws, safe_mode=safe_mode, inplace=inplace
    )

    if draws is None:
        size = 500
    elif isinstance(draws, int):
        size = draws
    elif isinstance(draws, np.ndarray):
        size = draws.size
    elif isinstance(draws, list):
        size = len(draws)
    else:
        raise ValueError("draws must be int, None, np.ndarray, or list")

    try:
        if inplace:
            assert "posterior_predictive" in cav_dt_copy
            assert cav_dt_copy.posterior_predictive.draw.size == size
        else:
            assert posterior_predictive is not None
            assert "posterior_predictive" not in cav_dt_copy
            assert posterior_predictive.posterior_predictive.draw.size == size
    except AssertionError:
        raise
