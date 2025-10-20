from pathlib import Path
import pytest

import jax
import numpy as np

import hssm
from hssm.likelihoods.rldm_optimized_abstraction import (
    make_rl_logp_func,
    make_rldm_logp_op,
    compute_v_subject_wise,
    _validate_columns
)

hssm.set_floatX("float32")

DECIMAL = 2


@pytest.fixture
def fixture_path():
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def rldm_setup(fixture_path):
    data = np.load(fixture_path / "rldm_data.npy", allow_pickle=True).item()["data"]
    participant_id = data["participant_id"].values
    trial = data["trial"].values
    subj = np.unique(participant_id).astype(np.int32)
    total_trials = trial.size

    data["rl_alpha"] = np.ones(total_trials, dtype=np.float32) * 0.60
    data["scaler"] = np.ones(total_trials, dtype=np.float32) * 3.2
    data["action"] = data.response.values
    data["feedback"] = data.pop("feedback")

    dist_params = ["rl_alpha", "scaler"]
    extra_fields = ["action", "feedback"]

    compute_v_subject_wise.inputs = ["rl_alpha", "scaler", "action", "feedback"]
    compute_v_subject_wise.outputs = ["v"]

    logp_fn = make_rl_logp_func(
        compute_v_subject_wise,
        n_participants=len(subj),
        n_trials=total_trials // len(subj),
        data_cols=list(data.columns),
        dist_params=dist_params,
        extra_fields=extra_fields,
    )

    return {
        "data": data,
        "values": data.values,
        "logp_fn": logp_fn,
        "total_trials": total_trials,
    }


class TestValidateColumns:
    def test_passes_when_all_present(self):
        data_cols = ["rl_alpha", "scaler", "response", "feedback", "a", "z"]
        dist_params = ["rl_alpha", "scaler"]
        extra_fields = ["response", "feedback"]
        assert _validate_columns(data_cols, dist_params, extra_fields) is None

    def test_missing_dist_param_raises(self):
        data_cols = ["scaler", "response", "feedback"]
        dist_params = ["rl_alpha", "scaler"]
        with pytest.raises(ValueError) as exc:
            _validate_columns(data_cols, dist_params, extra_fields=["response"])
        msg = str(exc.value)
        assert "rl_alpha" in msg
        assert "missing" in msg.lower()

    def test_missing_extra_field_raises(self):
        data_cols = ["rl_alpha", "scaler", "feedback"]
        extra_fields = ["response", "feedback"]
        with pytest.raises(ValueError) as exc:
            _validate_columns(data_cols, dist_params=["rl_alpha"], extra_fields=extra_fields)
        msg = str(exc.value)
        assert "response" in msg
        assert "missing" in msg.lower()

    def test_no_params_lists(self):
        _validate_columns(data_cols=["anything"])
        _validate_columns(data_cols=["anything"], dist_params=[], extra_fields=[])


class TestRldmLikelihoodAbstraction:
    def test_make_rl_logp_func(self, rldm_setup):
        setup = rldm_setup
        logp_fn = setup["logp_fn"]
        data = setup["values"]
        total_trials = setup["total_trials"]
        drift_rates = logp_fn(data)
        assert drift_rates.shape[0] == total_trials
        np.testing.assert_allclose(drift_rates.sum(), -39.509395, rtol=1e-2)

        jitted_logp = jax.jit(logp_fn)
        jax_ll = jitted_logp(data)
        assert np.all(jax_ll == drift_rates)
