"""Models behind the recorded trace fixtures, and the recipe that regenerates them.

The ``.nc`` traces under ``tests/fixtures/`` must match the PyMC graph that the
installed bambi builds for these models: bambi's ``predict`` and
``compute_log_likelihood`` read every free RV from ``posterior`` (#1336). When a
bambi upgrade renames a free RV or a dim, refresh the traces with::

    uv run python -m tests.trace_fixtures

``tests/conftest.py`` builds its fixture models from the same functions, so the
model a trace is loaded into is always the model it was sampled from.
"""

from pathlib import Path

import pandas as pd

import hssm

FIXTURES = Path(__file__).parent / "fixtures"
SEED = 2026


def build_cav_reg_model(data: pd.DataFrame) -> hssm.HSSM:
    """Return the regression DDM behind ``cavanagh_idata.nc``."""
    return hssm.HSSM(
        data=data,
        include=[
            {
                "name": "v",
                "prior": {
                    "Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                    "theta": {"name": "Normal", "mu": 0.0, "sigma": 1.0},
                },
                "formula": "v ~ theta + (1|participant_id)",
                "link": "identity",
            },
        ],
    )


def _cartoon_include(drift: str) -> list[dict]:
    return [
        {
            "name": drift,
            "prior": {"Intercept": {"name": "Normal", "mu": 0.0, "sigma": 1.5}},
            "formula": f"{drift} ~ 1 + stim",
            "link": "identity",
        },
        {
            "name": "a",
            "prior": {"Intercept": {"name": "Normal", "mu": 1.5, "sigma": 0.5}},
            "formula": "a ~ 1 + (1|participant_id)",
            "link": "identity",
        },
    ]


def build_cav_cartoon_model(data: pd.DataFrame) -> hssm.HSSM:
    """Return the DDM behind ``idata_cavanagh_cartoon.nc``."""
    return hssm.HSSM(
        model="ddm", data=data, include=_cartoon_include("v"), p_outlier=0.00
    )


def build_race_cartoon_model(data: pd.DataFrame) -> hssm.HSSM:
    """Return the 4-choice race model behind ``test_idata_race.nc``."""
    return hssm.HSSM(
        model="race_no_bias_angle_4",
        data=data,
        include=_cartoon_include("v0"),
        p_outlier=0.00,
    )


def _sample(model: hssm.HSSM, draws: int, tune: int):
    return model.sample(
        sampler="numpyro",
        chains=2,
        cores=1,
        chain_method="vectorized",
        draws=draws,
        tune=tune,
        random_seed=SEED,
        idata_kwargs={"log_likelihood": False},
    )


def main() -> None:
    """Resample every trace fixture and overwrite it in place."""
    cavanagh_test = pd.read_csv(FIXTURES / "cavanagh_theta_test.csv")

    # Draw count is pinned by tests that subsample it (`test__random_sample`,
    # `test_sample_posterior_predictive`); the predictive group feeds the
    # quantile-probability and predictive plot tests.
    cav_reg = build_cav_reg_model(cavanagh_test)
    traces = _sample(cav_reg, draws=200, tune=200)
    cav_reg.sample_posterior_predictive(traces, safe_mode=False)
    traces.to_netcdf(FIXTURES / "cavanagh_idata.nc")

    cav_cartoon = build_cav_cartoon_model(cavanagh_test)
    _sample(cav_cartoon, draws=100, tune=200).to_netcdf(
        FIXTURES / "idata_cavanagh_cartoon.nc"
    )

    race = build_race_cartoon_model(pd.read_csv(FIXTURES / "data_race.csv"))
    _sample(race, draws=100, tune=200).to_netcdf(FIXTURES / "test_idata_race.nc")


if __name__ == "__main__":
    main()
