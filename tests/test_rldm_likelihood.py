from pathlib import Path
import pytest

import jax
import numpy as np

import hssm
from hssm.likelihoods.rldm import make_logp_func, make_rldm_logp_op

hssm.set_floatX("float32")

DECIMAL = 4


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures"


def test_make_logp_func(fixture_path):
    """Test the JAX log-likelihood function for the RLDM model."""
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    feedback = data["feedback"].values

    subj = np.unique(participant_id).astype(np.int32)
    n_trials = trial.size // len(subj)

    rl_alpha = np.ones(n_trials) * 0.60
    scaler = np.ones(n_trials) * 3.2
    a = np.ones(n_trials) * 1.2
    z = np.ones(n_trials) * 0.1
    t = np.ones(n_trials) * 0.1
    theta = np.ones(n_trials) * 0.1

    logp = make_logp_func(n_participants=len(subj), n_trials=n_trials)
    jitted_logp = jax.jit(logp)

    jax_LL = jitted_logp(
        data[["rt", "response"]].values,
        rl_alpha,
        scaler,
        a,
        z,
        t,
        theta,
        participant_id,
        trial,
        feedback,
    )

    assert jax_LL.shape == (len(subj) * n_trials,)

    np.testing.assert_almost_equal(
        jax_LL.sum(),
        -6879.15262966,
        decimal=DECIMAL,
    )


def test_make_rldm_logp_op(fixture_path):
    """Test the JAX log-likelihood function for the RLDM model."""
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    feedback = data["feedback"].values

    subj = np.unique(participant_id).astype(np.int32)
    n_trials = trial.size // len(subj)

    rl_alpha = np.ones(n_trials) * 0.60
    scaler = np.ones(n_trials) * 3.2
    a = np.ones(n_trials) * 1.2
    z = np.ones(n_trials) * 0.1
    t = np.ones(n_trials) * 0.1
    theta = np.ones(n_trials) * 0.1

    logp_op = make_rldm_logp_op(
        n_participants=len(subj),
        n_trials=n_trials,
    )

    jax_LL = logp_op(
        data.loc[:, ["rt", "response"]].values,
        rl_alpha,
        scaler,
        a,
        z,
        t,
        theta,
        participant_id,
        trial,
        feedback,
    )

    np.testing.assert_almost_equal(jax_LL.sum().eval(), -6879.1523, decimal=DECIMAL)
