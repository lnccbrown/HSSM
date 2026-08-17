"""Sampler capability contract for the analytical JEAM likelihood."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

pytest.importorskip("jeam")

import hssm
from hssm.integrations.jeam import simulate_circular_diffusion
from hssm.modelconfig import get_default_model_config

PARAMETERS = ("v_x", "v_y", "a", "t")
NUTS_STATISTICS = {"acceptance_rate", "diverging", "n_steps", "step_size", "tree_depth"}


@pytest.fixture(scope="module")
def circular_data() -> pd.DataFrame:
    """Return one deterministic dataset shared by all sampler probes."""
    observations = simulate_circular_diffusion(
        theta=np.array([0.70, -0.35, 0.80, 0.08]),
        random_state=1947,
        n_replicas=24,
    )
    return pd.DataFrame(observations, columns=["rt", "response"])


def _make_analytical_model(data: pd.DataFrame) -> hssm.HSSM:
    """Construct an ordinary HSSM with JEAM's analytical JAX likelihood."""
    return hssm.HSSM(
        data=data,
        model="circular_diffusion",
        loglik_kind="analytical",
        p_outlier=None,
    )


def test_circular_config_declares_verified_sampler_combinations():
    """Each likelihood route should advertise only its verified samplers."""
    config = get_default_model_config("circular_diffusion")

    assert config["likelihoods"]["blackbox"]["supported_samplers"] == ("pymc",)
    assert config["likelihoods"]["analytical"]["supported_samplers"] == (
        "pymc",
        "numpyro",
    )


@pytest.mark.parametrize("sampler", ["blackjax", "nutpie", "laplace"])
def test_unverified_analytical_samplers_fail_before_backend_dispatch(
    circular_data: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    sampler: str,
):
    """Unverified JEAM combinations must fail before Bambi starts inference."""
    model = _make_analytical_model(circular_data)

    def unexpected_fit(**kwargs):
        pytest.fail(f"Bambi.fit was called for unsupported sampler {sampler!r}.")

    monkeypatch.setattr(model.model, "fit", unexpected_fit)

    with pytest.raises(
        ValueError,
        match=rf"Sampler {sampler!r} is not supported.*circular_diffusion.*analytical",
    ):
        model.sample(
            sampler=sampler,
            chains=1,
            cores=1,
            tune=1,
            draws=1,
            progressbar=False,
        )


@pytest.mark.slow
@pytest.mark.parametrize("sampler", ["pymc", "numpyro"])
def test_verified_nuts_backends_compile_and_return_nuts_statistics(
    circular_data: pd.DataFrame,
    sampler: str,
):
    """The two supported NUTS backends must use finite gradients and NUTS steps."""
    model = _make_analytical_model(circular_data)
    transformed_init = pm.initial_point.make_initial_point_fn(
        model=model.pymc_model,
        overrides=model.initvals,
        return_transformed=True,
    )(None)
    gradient = np.asarray(model.pymc_model.compile_dlogp()(transformed_init))

    assert gradient.shape == (len(PARAMETERS),)
    assert np.all(np.isfinite(gradient))
    assert np.any(gradient != 0.0)

    traces = model.sample(
        sampler=sampler,
        chains=1,
        cores=1,
        tune=10,
        draws=10,
        random_seed=3101,
        progressbar=False,
        idata_kwargs={"log_likelihood": False},
    )

    assert isinstance(traces, xr.DataTree)
    assert set(PARAMETERS) <= set(traces["posterior"].data_vars)
    for parameter in PARAMETERS:
        values = np.asarray(traces["posterior"][parameter])
        assert values.shape == (1, 10)
        assert np.all(np.isfinite(values))

    sample_statistics = set(traces["sample_stats"].data_vars)
    assert NUTS_STATISTICS <= sample_statistics
    assert {"nstep_in", "nstep_out"}.isdisjoint(sample_statistics)
    if sampler == "pymc":
        assert "process_time_diff" in sample_statistics
    else:
        assert "process_time_diff" not in sample_statistics
