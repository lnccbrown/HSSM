"""RLSSM with the analytical LBA (lba2, lba3, lba4) as decision process.

The learner is ssms's ``RescorlaWagnerRaceDrifts``: one drift ``v_i = scaler * q_i``
per arm, responses coded ``0, ..., n - 1``. ssms ships no RW + LBA preset, so the
model is an ``ssms.rl.ModelConfig`` passed through ``RLSSMConfig.from_ssms_model``.
"""

from copy import deepcopy

import arviz as az
import jax
import numpy as np
import pymc as pm
import pytensor
import pytest
from ssms import rl
from ssms.rl.env import Bandit
from ssms.rl.learning import RescorlaWagnerRaceDrifts

import hssm
from hssm.likelihoods.analytical import logp_lba_jax
from hssm.rl import registry

TRUTH = {"rl_alpha": 0.3, "scaler": 3.0, "A": 0.3, "b": 0.6}


@pytest.fixture(scope="module", autouse=True)
def _float64_precision_guard():
    """Pin float64 for this module and restore prior precision settings."""
    prev_floatx = pytensor.config.floatX
    prev_jax_x64 = jax.config.read("jax_enable_x64")
    hssm.set_floatX("float64")
    yield
    pytensor.config.floatX = prev_floatx
    jax.config.update("jax_enable_x64", prev_jax_x64)


@pytest.fixture(autouse=True)
def isolated_registries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate global registries and start every test with an empty logp cache."""
    monkeypatch.setattr(registry, "_SSM_REGISTRY", deepcopy(registry._SSM_REGISTRY))
    monkeypatch.setattr(registry, "_RLSSM_REGISTRY", deepcopy(registry._RLSSM_REGISTRY))
    monkeypatch.setattr(registry, "_SSM_LOGP_CACHE", {})


def _rw_lba(n: int) -> rl.ModelConfig:
    """Return an n-armed Bernoulli bandit with an RW learner and an n-choice LBA."""
    return rl.ModelConfig(
        model_name=f"{n}AB_RW_LBA{n}",
        description=f"{n}-armed bandit, Rescorla-Wagner learner, LBA{n}.",
        decision_process=f"lba{n}",
        learning_process=RescorlaWagnerRaceDrifts(n_actions=n, initial_q=0.5),
        task_environment=Bandit.bernoulli(
            probabilities=list(np.linspace(0.8, 0.2, n)),
            response_labels=list(range(n)),
        ),
    )


def _simulate(n, n_trials, n_participants, seed):
    # ssms's LBA simulator draws from NumPy's global RNG, not random_state
    # (ssm-simulators#399).
    np.random.seed(seed)
    return rl.Simulator(_rw_lba(n)).simulate(
        theta=TRUTH,
        n_trials=n_trials,
        n_participants=n_participants,
        random_state=seed,
    )


def _lba_model(n, data, **kwargs):
    config = hssm.rl.RLSSMConfig.from_ssms_model(
        _rw_lba(n), decision_process_loglik_kind="analytical"
    )
    return hssm.rl.RLSSM(data, model_config=config, p_outlier=None, **kwargs)


def _reference_logp(n, data, rl_alpha, scaler, A, b):
    """Sum of logp: a NumPy Rescorla-Wagner learner plus `logp_lba_jax`."""
    drifts = []
    for _, trials in data.groupby("participant_id", sort=False):
        q = np.full(n, 0.5)
        for response, reward in zip(trials["response"], trials["feedback"]):
            drifts.append(scaler * q)
            q[int(response)] += rl_alpha * (reward - q[int(response)])
    v = np.array(drifts).T
    obs = data[["rt", "response"]].to_numpy()
    return float(np.sum(logp_lba_jax(obs, A, b, *v)))


def _observed_logp(model):
    """Compile the observed log-likelihood as a function of the named parameters."""
    pymc_model = model.pymc_model
    obs = pymc_model.observed_RVs[0]
    logp = pm.logp(obs, pymc_model.rvs_to_values[obs]).sum()
    fn = pm.compile(inputs=[pymc_model[name] for name in TRUTH], outputs=logp)
    return lambda point: fn(*(np.asarray(point[name]) for name in TRUTH))


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("point", [TRUTH, {**TRUTH, "rl_alpha": 0.6, "A": 0.45}])
def test_observed_logp_matches_reference(n, point):
    """The model's likelihood is the RW race learner composed with the LBA."""
    data = _simulate(n, n_trials=40, n_participants=2, seed=n)

    logp = _observed_logp(_lba_model(n, data))(point)

    np.testing.assert_allclose(logp, _reference_logp(n, data, **point), atol=1e-9)


def test_start_satisfies_b_greater_than_A():
    """The model starts at ssms's defaults (A 0.3, b 0.5), not the box midpoints.

    Both midpoints are 0.5, where the LBA's log-likelihood is -inf.
    """
    data = _simulate(3, n_trials=40, n_participants=2, seed=0)

    initvals = _lba_model(3, data).initvals
    user = _lba_model(
        3,
        data,
        include=[
            {
                "name": "b",
                "prior": {
                    "name": "Uniform",
                    "lower": 0.0,
                    "upper": 1.0,
                    "initval": 0.9,
                },
            }
        ],
    ).initvals

    np.testing.assert_allclose([initvals["A"], initvals["b"]], [0.3, 0.5], atol=0.02)
    assert np.isfinite(_observed_logp(_lba_model(3, data))({**TRUTH, **initvals}))
    np.testing.assert_allclose(user["b"], 0.9, atol=0.02)


@pytest.mark.slow
def test_parameter_recovery():
    """RW + LBA3 recovers the generating parameters and samples without divergences.

    The (A, b, v) scale ridge mixes slowly, so the r-hat bound is a smoke-test 1.1,
    and the 3-SD tolerance targets gross errors only.
    """
    data = _simulate(3, n_trials=150, n_participants=4, seed=1)

    idata = _lba_model(3, data).sample(
        sampler="numpyro", chains=2, tune=400, draws=400, random_seed=3
    )

    posterior = idata["posterior"]
    for name, value in TRUTH.items():
        draws = np.asarray(posterior[name]).ravel()
        assert abs(draws.mean() - value) < 3 * draws.std(), name
    assert int(idata["sample_stats"]["diverging"].sum()) == 0
    rhat = az.rhat(posterior.dataset, var_names=list(TRUTH))
    assert float(rhat.to_array().max()) <= 1.1
