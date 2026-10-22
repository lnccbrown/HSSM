"""Choice-only (softmax inverse temperature) simulation and lapse (#925)."""

import numpy as np
import pandas as pd
import pytest
from scipy.special import softmax

import hssm

hssm.set_floatX("float32", update_jax=True)

N_DRAWS = 20_000


def _model(n_choices, **kwargs):
    choices = [-1, 1] if n_choices == 2 else list(range(n_choices))
    data = pd.DataFrame({"response": np.random.default_rng(0).choice(choices, size=50)})
    return hssm.HSSM(data=data, model=f"softmax_inv_temperature_{n_choices}", **kwargs)


@pytest.mark.parametrize(
    ("n_choices", "theta"),
    [(2, [2.0, 0.7]), (3, [2.0, -1.0, 0.5]), (4, [1.5, -1.0, 0.5, 1.0])],
)
def test_rv_matches_softmax(n_choices, theta):
    """Simulated choice frequencies follow softmax(beta * [0, logit1, ...])."""
    model = _model(n_choices, p_outlier=None)
    rv = model.model_distribution.rv_op

    draws = rv.rng_fn(np.random.default_rng(1), *theta, N_DRAWS)

    assert draws.shape == (N_DRAWS,)
    beta, *logits = theta
    expected = softmax(beta * np.array([0.0, *logits]))
    observed = np.array([np.mean(draws == c) for c in model.choices])
    np.testing.assert_allclose(observed, expected, atol=0.015)


def test_rv_trialwise_params_keep_trial_shape():
    """Vector parameters (regressions) produce one choice per trial."""
    rv = _model(3, p_outlier=None).model_distribution.rv_op
    beta = np.linspace(0.5, 3.0, 7)

    draws = rv.rng_fn(np.random.default_rng(2), beta, -1.0, 0.5, None)

    assert draws.shape == (7,)
    assert set(np.unique(draws)) <= {0, 1, 2}


def test_rv_lapse_draws_uniform_choices():
    """With p_outlier=1 every trial is a lapse: choices are uniform."""
    model = _model(3)  # default p_outlier with numeric lapse 1 / n_choices
    rv = model.model_distribution.rv_op

    # A near-deterministic softmax, so any spread comes from the lapse.
    draws = rv.rng_fn(np.random.default_rng(3), 20.0, -5.0, -5.0, 1.0, N_DRAWS)

    assert draws.shape == (N_DRAWS,)
    observed = np.array([np.mean(draws == c) for c in model.choices])
    np.testing.assert_allclose(observed, 1 / 3, atol=0.015)


def test_lapse_logp_is_a_mixture_of_probabilities():
    """logp = log((1 - p) * softmax + p / n_choices), never above 0."""
    model = _model(2, p_outlier=0.5)
    point = model.pymc_model.initial_point()

    logp = model.pymc_model.compile_logp(sum=False)(point)[-1]

    # The initial point has logit1 = 0, so the softmax is 1/2 for each choice.
    np.testing.assert_allclose(logp, np.log(0.5), rtol=1e-5)


def test_prior_predictive_draws_valid_choices():
    """Prior predictive sampling runs end to end for choice-only models."""
    model = _model(2)

    prior = model.sample_prior_predictive(draws=5)

    response = prior["prior_predictive"]["response"].values
    assert set(np.unique(response)) <= {-1, 1}


@pytest.mark.slow
def test_posterior_predictive_draws_valid_choices():
    """Posterior predictive sampling runs end to end for choice-only models."""
    model = _model(2)
    model.sample(draws=10, tune=10, chains=1, cores=1, progressbar=False)

    model.sample_posterior_predictive(draws=5)

    response = model.traces["posterior_predictive"]["response"]
    assert response.dims == ("chain", "draw", "__obs__")
    assert set(np.unique(response.values)) <= {-1, 1}


def test_choice_only_ssms_model_keeps_only_responses():
    """A choice-only view of an ssms model simulates through ssms, minus rts."""
    data = pd.DataFrame({"rt": np.ones(10), "response": np.tile([-1.0, 1.0], 5)})
    model = hssm.HSSM(
        data=data,
        model="ddm",
        model_config={"response": ["response"]},
        p_outlier=None,
    )

    draws = model.model_distribution.rv_op.rng_fn(
        np.random.default_rng(4), 0.5, 1.5, 0.5, 0.3, 100
    )

    assert draws.shape == (100,)
    assert set(np.unique(draws)) == {-1, 1}


@pytest.mark.parametrize(
    "lapse",
    [0.5, {"name": "Uniform", "lower": 0.0, "upper": 20.0}],
    ids=["non_uniform_probability", "rt_prior"],
)
def test_non_uniform_lapse_is_rejected(lapse):
    """Choice-only lapses are uniform over choices, so simulation matches logp."""
    with pytest.raises(ValueError, match="uniform lapse"):
        _model(3, lapse=lapse)


def test_explicit_uniform_lapse_is_accepted():
    """Passing the uniform lapse explicitly is the same as the default."""
    assert _model(3, lapse=1 / 3).lapse == pytest.approx(1 / 3)
