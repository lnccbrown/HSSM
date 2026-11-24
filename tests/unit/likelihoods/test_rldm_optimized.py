from pathlib import Path
import pytest

import jax
import numpy as np

import hssm
from hssm.likelihoods import rldm_optimized
from hssm.likelihoods.rldm_optimized import make_rldm_logp_func, make_rldm_logp_op

hssm.set_floatX("float32")

DECIMAL = 2

# The tests pass the local angle.onnx fixture, so a Hub download means `model` was
# ignored (the Hub copy is byte-identical and would otherwise pass silently).
pytestmark = pytest.mark.usefixtures("no_hf_hub_download")


@pytest.fixture
def fixture_path():
    return Path(__file__).parents[2] / "fixtures"


def test_make_rldm_logp_func(fixture_path):
    """Test the JAX log-likelihood function for the RLDM model."""
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    feedback = data["feedback"].values

    subj = np.unique(participant_id).astype(np.int32)
    total_trials = trial.size

    rl_alpha = np.ones(total_trials) * 0.60
    scaler = np.ones(total_trials) * 3.2
    a = np.ones(total_trials) * 1.2
    z = np.ones(total_trials) * 0.1
    t = np.ones(total_trials) * 0.1
    theta = np.ones(total_trials) * 0.1

    logp = make_rldm_logp_func(
        n_participants=len(subj),
        n_trials=total_trials // len(subj),
        model=fixture_path / "angle.onnx",
    )
    jitted_logp = jax.jit(logp)
    jax_ll = jitted_logp(
        data[["rt", "response"]].values,
        rl_alpha,
        scaler,
        a,
        z,
        t,
        theta,
        feedback,
    )

    assert jax_ll.shape[0] == total_trials
    np.testing.assert_allclose(jax_ll.sum(), -6879.1523, rtol=1e-2)


def test_make_rldm_logp_op(fixture_path):
    """Test the JAX log-likelihood function for the RLDM model."""
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    feedback = data["feedback"].values

    subj = np.unique(participant_id).astype(np.int32)
    total_trials = trial.size

    rl_alpha = np.ones(total_trials) * 0.60
    scaler = np.ones(total_trials) * 3.2
    a = np.ones(total_trials) * 1.2
    z = np.ones(total_trials) * 0.1
    t = np.ones(total_trials) * 0.1
    theta = np.ones(total_trials) * 0.1

    logp_op = make_rldm_logp_op(
        n_participants=len(subj),
        n_trials=total_trials // len(subj),
        n_params=6,
        model=fixture_path / "angle.onnx",
    )

    jax_ll = logp_op(
        data.loc[:, ["rt", "response"]].values,
        rl_alpha,
        scaler,
        a,
        z,
        t,
        theta,
        feedback,
    )

    np.testing.assert_almost_equal(jax_ll.sum().eval(), -6879.1523, decimal=DECIMAL)


@pytest.mark.parametrize(
    ("factory", "kwargs"),
    [(make_rldm_logp_func, {}), (make_rldm_logp_op, {"n_params": 6})],
)
def test_default_model_is_hub_angle(monkeypatch, factory, kwargs):
    """Without ``model``, the factories load ``angle.onnx`` from the Hub."""
    requested = []
    monkeypatch.setattr(
        rldm_optimized,
        "make_jax_matrix_logp_funcs_from_onnx",
        lambda model: requested.append(model),
    )
    factory(n_participants=1, n_trials=1, **kwargs)
    assert requested == ["angle.onnx"]
