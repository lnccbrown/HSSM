from pathlib import Path
import pytest

import jax
import numpy as np

import hssm
from hssm.likelihoods.rldm_optimized import make_rldm_logp_func, make_rldm_logp_op

hssm.set_floatX("float32")

DECIMAL = 2


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures"


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
        n_participants=len(subj), n_trials=total_trials // len(subj)
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
