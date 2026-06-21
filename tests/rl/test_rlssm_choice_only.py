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
    assert "q0" not in model.params  # computed inside Op, not a free param
    assert "q1" not in model.params  # computed inside Op, not a free param


def test_rlssm_softmax_no_rt_column(choice_only_data) -> None:
    """Choice-only RLSSM should not require an 'rt' column in the data."""
    assert "rt" not in choice_only_data.columns
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    assert model.model is not None


def test_rlssm_softmax_invalid_response_rejected() -> None:
    """Response values outside {0, 1} must be rejected at construction time.

    The softmax logp and the Q-value scan index per-action arrays by the integer
    response, so an out-of-range value such as -1 or 2 would silently produce a
    wrong likelihood (or corrupt Q-value updates via out-of-bounds indexing).
    ``_post_check_data_sanity`` must catch these before any JAX computation runs.
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


def _logp_at_initial_point(model: RLSSM) -> float:
    """Compile and evaluate the model logp at its initial point."""
    with model.pymc_model:
        ip = model.pymc_model.initial_point()
        return float(model.pymc_model.compile_logp()(ip))


@pytest.mark.parametrize(
    "label_map, choices",
    [
        ({0.0: 1.0, 1.0: 2.0}, [1, 2]),  # shifted labels
        ({0.0: -1.0, 1.0: 1.0}, [-1, 1]),  # signed labels (HSSM softmax convention)
    ],
)
def test_rlssm_softmax_custom_choice_labels_behave_like_binary_actions(
    label_map, choices
) -> None:
    """Arbitrary integer ``choices`` should reduce to the internal 0/1 actions.

    The public RLSSM API accepts ``choices`` overrides; the response column is
    remapped to 0-based action indices, so relabeling the two valid choices must
    not change the log-likelihood relative to the canonical ``[0, 1]`` coding.
    """
    rng = np.random.default_rng(7)
    n_participants, n_trials = 4, 12
    n_total = n_participants * n_trials

    binary_data = pd.DataFrame(
        {
            "participant_id": np.repeat(np.arange(n_participants), n_trials),
            "response": rng.integers(0, 2, size=n_total).astype(float),
            "feedback": rng.integers(0, 2, size=n_total).astype(float),
        }
    )
    relabeled_data = binary_data.copy()
    relabeled_data["response"] = relabeled_data["response"].replace(label_map)

    default_model = RLSSM(data=binary_data, model="2AB_RescorlaWagner_Softmax")
    relabeled_model = RLSSM(
        data=relabeled_data,
        model="2AB_RescorlaWagner_Softmax",
        choices=choices,
    )

    default_lp = _logp_at_initial_point(default_model)
    relabeled_lp = _logp_at_initial_point(relabeled_model)

    assert np.isclose(default_lp, relabeled_lp), (
        "Relabeling the two valid choices should not change the log-likelihood. "
        f"Expected matching values, got {default_lp} and {relabeled_lp}."
    )


def test_rlssm_softmax_choice_remap_respects_declared_order() -> None:
    """Remapping uses declared ``choices`` order: choices=[2, 1] maps 2->0, 1->1."""
    rng = np.random.default_rng(11)
    n_participants, n_trials = 4, 12
    n_total = n_participants * n_trials
    base_response = rng.integers(0, 2, size=n_total).astype(float)
    feedback = rng.integers(0, 2, size=n_total).astype(float)
    participant_id = np.repeat(np.arange(n_participants), n_trials)

    canonical = pd.DataFrame(
        {
            "participant_id": participant_id,
            "response": base_response,
            "feedback": feedback,
        }
    )
    # Map 0->2 and 1->1 so that declaring choices=[2, 1] inverts back to 0/1.
    reordered = canonical.copy()
    reordered["response"] = np.where(base_response == 0.0, 2.0, 1.0)

    canonical_lp = _logp_at_initial_point(
        RLSSM(data=canonical, model="2AB_RescorlaWagner_Softmax")
    )
    reordered_lp = _logp_at_initial_point(
        RLSSM(data=reordered, model="2AB_RescorlaWagner_Softmax", choices=[2, 1])
    )
    assert np.isclose(canonical_lp, reordered_lp)


def test_rlssm_softmax_remap_does_not_mutate_user_dataframe() -> None:
    """The remap must operate on HSSM's copy, leaving the user's frame untouched."""
    rng = np.random.default_rng(3)
    n_total = 40
    data = pd.DataFrame(
        {
            "participant_id": np.repeat(np.arange(4), 10),
            "response": rng.choice([1.0, 2.0], size=n_total),
            "feedback": rng.integers(0, 2, size=n_total).astype(float),
        }
    )
    original_response = data["response"].copy()
    RLSSM(data=data, model="2AB_RescorlaWagner_Softmax", choices=[1, 2])
    pd.testing.assert_series_equal(data["response"], original_response)


def test_rlssm_ddm_rejects_non_zero_based_choices() -> None:
    """SSM-backed RLSSMs must reject custom labels (response is dual-use)."""
    rng = np.random.default_rng(5)
    n_total = 40
    data = pd.DataFrame(
        {
            "participant_id": np.repeat(np.arange(4), 10),
            "rt": rng.uniform(0.3, 1.5, size=n_total),
            "response": rng.choice([1.0, 2.0], size=n_total),
            "feedback": rng.integers(0, 2, size=n_total).astype(float),
        }
    )
    with pytest.raises(ValueError, match="not supported for"):
        RLSSM(data=data, model="2AB_RescorlaWagner_DDM", choices=[1, 2])


def test_rlssm_softmax_warns_when_a_declared_choice_is_missing() -> None:
    """Choice-only RLSSM should warn when one declared response never appears."""
    rng = np.random.default_rng(11)
    n_participants, n_trials = 4, 12
    data = pd.DataFrame(
        {
            "participant_id": np.repeat(np.arange(n_participants), n_trials),
            "response": np.zeros(n_participants * n_trials, dtype=float),
            "feedback": rng.integers(0, 2, size=n_participants * n_trials).astype(
                float
            ),
        }
    )

    with pytest.warns(
        UserWarning,
        match=r"You set choices to be \(0, 1\), but \[1\] are missing from your dataset\.",
    ):
        RLSSM(data=data, model="2AB_RescorlaWagner_Softmax")


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


@pytest.fixture(scope="module")
def choice_only_data_3afc() -> pd.DataFrame:
    """Synthetic balanced-panel 3-alternative choice-only dataset (no RT)."""
    rng = np.random.default_rng(99)
    n_participants, n_trials = 5, 20
    n_total = n_participants * n_trials
    return pd.DataFrame(
        {
            "participant_id": np.repeat(np.arange(n_participants), n_trials),
            "response": rng.integers(0, 3, size=n_total).astype(float),
            "feedback": rng.integers(0, 2, size=n_total).astype(float),
        }
    )


def test_rlssm_softmax_3afc_instantiates_and_evaluates(choice_only_data_3afc) -> None:
    """The registered 3-action softmax model builds and yields valid logp.

    Exercises the N-action generalisation end to end: a single Q-value scan
    produces q0/q1/q2, the generic softmax consumes them, and per-trial
    log-probabilities are finite and non-positive.
    """
    model = RLSSM(data=choice_only_data_3afc, model="3AB_RescorlaWagner_Softmax")
    assert model.model_config.is_choice_only
    assert model.model_config.choices == (0, 1, 2)
    # q0/q1/q2 are computed inside the Op, not free parameters.
    assert {"rl_alpha", "beta"}.issubset(set(model.params))
    assert not ({"q0", "q1", "q2"} & set(model.params))

    with model.pymc_model:
        ip = model.pymc_model.initial_point()
        lp = model.pymc_model.compile_logp()(ip)
        parts = model.pymc_model.compile_logp(sum=False)(ip)
    assert np.isfinite(lp), f"Expected a finite logp, got {lp}"
    all_logp = np.concatenate([np.atleast_1d(x) for x in parts])
    assert np.all(all_logp <= 1e-6)


@pytest.mark.slow
def test_rlssm_softmax_sample_smoke(choice_only_data) -> None:
    """Minimal sampling run should return an InferenceData object."""
    model = RLSSM(data=choice_only_data, model="2AB_RescorlaWagner_Softmax")
    trace = model.sample(
        draws=100, tune=200, chains=2, cores=1, sampler="numpyro", target_accept=0.9
    )
    assert trace is not None
