from pathlib import Path

from matplotlib.pyplot import subplots_adjust
import pytest

import numpy as np

from hssm.likelihoods.rldm import logp


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures"


def test_rldm_logp(fixture_path):
    """Test the JAX log-likelihood function for the RLDM model."""
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    feedback = data["feedback"].values

    print(data.head())
    print(data.shape)

    subj = np.unique(participant_id).astype(np.int32)
    num_subj = len(subj)
    ntrials = len(trial)

    rl_alpha = np.ones(ntrials) * 0.60
    scaler = np.ones(ntrials) * 3.2
    a = np.ones(ntrials) * 1.2
    z = np.ones(ntrials) * 0.1
    t = np.ones(ntrials) * 0.1
    theta = np.ones(ntrials) * 0.1

    jax_LL = logp(
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

    np.testing.assert_almost_equal(
        jax_LL.sum(),
        -27090.58470433,
    )
