"""Every declared `jax_loglik` matches the PyTensor `loglik` of its likelihood entry."""

from typing import get_args

import jax
import numpy as np
import pytensor
import pytensor.tensor as pt
import pytest
from ssms.config import model_config as ssms_model_config

import hssm
from hssm._types import SupportedModels
from hssm.config import Config
from hssm.likelihoods.analytical import logp_ddm, logp_ddm_jax
from hssm.modelconfig import get_default_model_config

DECLARED = [
    (model, kind)
    for model in get_args(SupportedModels)
    for kind, entry in get_default_model_config(model)["likelihoods"].items()
    if entry.get("jax_loglik") is not None
]


@pytest.fixture(scope="module", autouse=True)
def _float64_precision_guard():
    """Pin float64 for this module and restore prior precision settings."""
    prev_floatx = pytensor.config.floatX
    prev_jax_x64 = jax.config.read("jax_enable_x64")
    hssm.set_floatX("float64")
    yield
    pytensor.config.floatX = prev_floatx
    jax.config.update("jax_enable_x64", prev_jax_x64)


def test_ddm_declares_a_jax_loglik():
    """The analytical DDM is among the declared entries; plain HSSM is unchanged."""
    config = Config.from_defaults("ddm", "analytical")

    assert ("ddm", "analytical") in DECLARED
    assert config.loglik is logp_ddm
    assert config.backend is None
    assert config.jax_loglik is logp_ddm_jax


@pytest.mark.parametrize(("model", "kind"), DECLARED)
def test_jax_loglik_matches_loglik(model, kind):
    """Per-trial parity on simulated data at the simulator's default parameters."""
    mc = get_default_model_config(model)
    entry = mc["likelihoods"][kind]
    theta = dict(
        zip(
            ssms_model_config[model]["params"],
            ssms_model_config[model]["default_params"],
        )
    )
    data = hssm.simulate_data(model=model, theta=theta, size=500, random_state=0)
    obs = data[mc["response"]].to_numpy(dtype="float64")
    params = [np.full(len(obs), theta[name]) for name in mc["list_params"]]

    symbols = [pt.vector(name, dtype="float64") for name in mc["list_params"]]
    obs_symbol = pt.matrix("obs", dtype="float64")
    loglik = pytensor.function(
        [obs_symbol, *symbols], entry["loglik"](obs_symbol, *symbols)
    )

    np.testing.assert_allclose(
        np.asarray(entry["jax_loglik"](obs, *params)),
        loglik(obs, *params),
        rtol=0,
        atol=1e-10,
    )
