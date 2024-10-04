import bambi as bmb
import numpy as np
import pytest

from hssm import Prior
from hssm.param.simple_param import DefaultParam


def test_from_defaults():
    """Test that the from_defaults class method works correctly."""
    param = DefaultParam.from_defaults(
        "test", prior={"name": "Normal", "mu": 0, "sigma": 1}, bounds=(0, 1)
    )

    assert isinstance(param, DefaultParam)
    assert param.name == "test"
    assert param.prior == {"name": "Normal", "mu": 0, "sigma": 1}
    assert param.bounds == (0, 1)

    param.process_prior()
    assert isinstance(param.prior, bmb.Prior) and not isinstance(param.prior, Prior)
    assert param.prior.name == "Normal"
    assert param.prior.args["mu"] == 0
    assert param.prior.args["sigma"] == 1


@pytest.mark.parametrize(
    ("bounds", "prior"),
    [
        ((0, 1), {"name": "Uniform", "lower": 0, "upper": 1}),
        ((0, np.inf), {"name": "HalfNormal", "sigma": 2}),
        ((1, np.inf), {"name": "TruncatedNormal", "sigma": 2, "lower": 1}),
        ((-np.inf, 1), {"name": "TruncatedNormal", "sigma": 2, "upper": 1}),
        ((-np.inf, np.inf), {"name": "Normal", "mu": 0, "sigma": 2}),
    ],
)
def test_make_default_prior(bounds, prior):
    """Test that the make_default_prior method works correctly."""
    param = DefaultParam.from_defaults("test", prior=None, bounds=bounds)
    assert param.prior.name == prior.pop("name")
    for key, value in prior.items():
        assert param.prior.args[key] == value
