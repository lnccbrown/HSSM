"""Tests for the choice-only (softmax) RLSSM model.

Covers the 2AB_RescorlaWagner_Softmax model: config validation,
instantiation without an RT column, and PyMC model construction.
"""

from collections.abc import Generator

import numpy as np
import pandas as pd
import pytensor
import pytest

import hssm
from hssm.rl import RLSSM


@pytest.fixture(scope="module", autouse=True)
def _set_floatx_float32() -> Generator[None, None, None]:
    """Ensure float32 is used for this module's tests, then restore previous setting."""
    prev_floatx = pytensor.config.floatX
    hssm.set_floatX("float32", update_jax=True)
    try:
        yield
    finally:
        hssm.set_floatX(prev_floatx, update_jax=True)


@pytest.fixture(scope="module")
def choice_only_data() -> pd.DataFrame:
    """Synthetic balanced-panel choice-only dataset (no RT column).

    5 participants × 20 trials each, choices ∈ {0, 1}, feedback ∈ {0.0, 1.0}.
    """
    rng = np.random.default_rng(42)
    n_participants, n_trials = 5, 20
    n_total = n_participants * n_trials
    return pd.DataFrame(
        {
            "participant_id": np.repeat(np.arange(n_participants), n_trials),
            "response": rng.integers(0, 2, size=n_total).astype(float),
            "feedback": rng.integers(0, 2, size=n_total).astype(float),
        }
    )


def test_rlssm_softmax_instantiate(choice_only_data) -> None:
    """2AB_RescorlaWagner_Softmax should instantiate without error."""
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    assert isinstance(model, RLSSM)
    assert model.model_config.is_choice_only
    assert "rl_alpha" in model.params
    assert "beta" in model.params
    assert "q_diff" not in model.params  # computed inside Op, not a free param


def test_rlssm_softmax_no_rt_column(choice_only_data) -> None:
    """Choice-only RLSSM should not require an 'rt' column in the data."""
    assert "rt" not in choice_only_data.columns
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    assert model.model is not None


def test_rlssm_softmax_invalid_response_rejected() -> None:
    """Response values outside {0, 1} must be rejected at construction time.

    The softmax logp uses ``jnp.where(response == 1, ...)``, treating anything
    other than 1 as choice 0.  An invalid value such as -1 or 2 would therefore
    silently produce a wrong likelihood (and corrupt Q-value updates via negative
    array indexing).  ``_post_check_data_sanity`` must catch these before any JAX
    computation runs.
    """
    rng = np.random.default_rng(0)
    n_total = 50
    bad_data = pd.DataFrame(
        {
            "participant_id": np.repeat(np.arange(5), 10),
            "response": np.where(
                rng.random(n_total) < 0.1,
                -1,  # inject invalid responses
                rng.integers(0, 2, size=n_total),
            ).astype(float),
            "feedback": rng.integers(0, 2, size=n_total).astype(float),
        }
    )
    with pytest.raises(ValueError, match="Invalid responses"):
        RLSSM(data=bad_data, model="2AB_RescorlaWagner_Softmax")


def test_rlssm_softmax_non_integral_response_rejected() -> None:
    """Non-integer response values (e.g. 0.5) must be rejected before the int cast.

    Casting to int first would silently convert 0.5 → 0, which passes the
    membership check against choices=[0, 1] and then corrupts Q-value indexing
    in the JAX learning step.  ``_post_check_data_sanity`` must detect the
    fractional value and raise before any computation runs.
    """
    rng = np.random.default_rng(1)
    n_total = 50
    participant_ids = np.repeat(np.arange(5), 10)
    feedback = rng.integers(0, 2, size=n_total).astype(float)
    bad_responses = np.where(
        rng.random(n_total) < 0.1,
        0.5,  # inject non-integer responses that would cast to 0
        rng.integers(0, 2, size=n_total),
    ).astype(float)
    bad_data = pd.DataFrame(
        {
            "participant_id": participant_ids,
            "response": bad_responses,
            "feedback": feedback,
        }
    )
    with pytest.raises(ValueError, match="Non-integer response values"):
        RLSSM(data=bad_data, model="2AB_RescorlaWagner_Softmax")


def test_rlssm_softmax_pymc_model(choice_only_data) -> None:
    """pymc_model should be accessible after softmax model construction."""
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    assert model.pymc_model is not None


def test_rlssm_softmax_logp_evaluates(choice_only_data) -> None:
    """logp must evaluate to a finite scalar without shape errors.

    The HSSMRV has scalar output for choice-only models, so PyMC supplies
    observed data as a 1D (n_obs,) tensor.  The RL logp Op requires 2D input
    (data[:, col_idx]).  This test confirms the reshape is applied correctly.
    """
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    with model.pymc_model:
        ip = model.pymc_model.initial_point()
        lp = model.pymc_model.compile_logp()(ip)
    assert np.isfinite(lp), f"Expected a finite logp, got {lp}"


def test_rlssm_softmax_per_trial_logp_valid(choice_only_data) -> None:
    """Per-trial log-probabilities must be ≤ 0 (valid log-probabilities).

    When lapse is a float (set to 1/n_choices for choice-only models),
    make_distribution must treat it as a probability and pass log(lapse) to the
    lapse_func — not the raw float.  Passing 0.5 directly causes exp(0.5) ≈ 1.649
    in the lapse mixture, which can push the per-trial probability above 1 and
    yield a positive log-probability.
    """
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    with model.pymc_model:
        ip = model.pymc_model.initial_point()
        # compile_logp(sum=False) returns a list of per-component arrays
        parts = model.pymc_model.compile_logp(sum=False)(ip)
    all_logp = np.concatenate([np.atleast_1d(x) for x in parts])
    assert np.all(all_logp <= 1e-6), (
        f"Some log-probabilities are positive (> 0), indicating the "
        f"lapse mixture probability exceeded 1. Max: {all_logp.max():.4f}"
    )


@pytest.mark.slow
def test_rlssm_softmax_sample_smoke(choice_only_data) -> None:
    """Minimal sampling run should return an InferenceData object."""
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    trace = model.sample(
        draws=100, tune=200, chains=2, cores=1, sampler="numpyro", target_accept=0.9
    )
    assert trace is not None
