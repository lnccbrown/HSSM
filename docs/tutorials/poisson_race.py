"""Simulate and fit an analytical Poisson race model with HSSM.

This marimo notebook is the source of truth for the output-bearing Jupyter
artifact published by MkDocs. Run a quick local check with::

    uv run --group notebook --group docs marimo check --strict \
        docs/tutorials/poisson_race.py
    uv run --group notebook --group docs marimo export html --no-sandbox \
        docs/tutorials/poisson_race.py \
        --output /tmp/poisson-race.html --force

Regenerate the committed full-run artifact with::

    FULL_RUN=1 uv run --group notebook --group docs marimo export ipynb \
        --no-sandbox docs/tutorials/poisson_race.py \
        --output docs/tutorials/poisson_race.ipynb \
        --include-outputs --force
    uv run ruff format docs/tutorials/poisson_race.ipynb
"""

# ruff: noqa: B018, D401, E501, PLR1711  (generated marimo notebook: prose, display expressions, and bare returns)
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import os
    import warnings

    # This tutorial does not need a GPU. Keep Molab and headless documentation
    # builds away from host CUDA plugins before importing HSSM/JAX.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

    import arviz as az
    import marimo as mo
    import numpy as np

    import hssm

    hssm.set_floatX("float32")
    az.style.use("seaborn-v0_8-whitegrid")

    FULL_RUN = os.environ.get("FULL_RUN") == "1"
    return FULL_RUN, az, hssm, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Poisson race models

    This short tutorial shows how to:

    1. simulate synthetic reaction times with the Poisson race simulator from
       `ssm-simulators`;
    2. fit HSSM's analytical Poisson race likelihood; and
    3. compare the recovered posterior with the generating parameters.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulate with ssm-simulators

    `hssm.simulate_data` wraps the `ssm-simulators` Poisson race generator.
    The simulator and HSSM likelihood share the same accumulator-specific
    parameter names: `r1`, `r2`, `k1`, `k2`, and the non-decision time `t`.
    """)
    return


@app.cell
def _(hssm, mo):
    true_params = {
        "r1": 1.0,
        "r2": 5.0,
        "k1": 2.0,
        "k2": 2.0,
        "t": 0.25,
    }

    data = hssm.simulate_data(
        model="poisson_race",
        theta=true_params,
        size=500,
        random_state=123,
    )
    mo.md(data.head().to_markdown(index=False))
    return data, true_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit the Poisson race likelihood

    This model has no regression terms, so its positive simple-parameter priors
    come from HSSM's model configuration. The model-level `prior_settings`
    selector is not a prior dictionary: it accepts only `"safe"` or `None` and
    controls generated **regression-term** priors. To replace a prior here,
    provide it through a parameter specification such as `include=[...]` or a
    parameter keyword.

    Because this is a synthetic recovery check, we start NUTS adaptation at the
    known generating point. Initial values choose where adaptation begins; they
    do not change the posterior target. With real data, use scientifically
    plausible starts and diagnose multiple chains.
    """)
    return


@app.cell
def _(FULL_RUN, data, hssm, true_params):
    _draws = 3_000 if FULL_RUN else 250
    _tune = 3_000 if FULL_RUN else 250

    _poisson_model = hssm.HSSM(
        data=data,
        model="poisson_race",
        loglik_kind="analytical",
    )

    idata = _poisson_model.sample(
        draws=_draws,
        tune=_tune,
        chains=2,
        cores=1,
        target_accept=0.95,
        initvals=true_params,
        random_seed=123,
        progressbar=False,
    )
    return (idata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare posteriors against the ground truth

    ArviZ summarizes the marginal posterior distributions. Adding the
    generating values to the table and plot makes it easy to check whether the
    fitted uncertainty covers the parameters used to simulate the data. In the
    plot, solid red lines mark the generating values and dashed blue lines mark
    posterior means.
    """)
    return


@app.cell
def _(az, idata, mo, true_params):
    var_names = list(true_params)
    summary = az.summary(idata, var_names=var_names)
    summary["true_value"] = [true_params[name] for name in summary.index]
    mo.md(summary.to_markdown())
    return (var_names,)


@app.cell
def _(az, idata, np, true_params, var_names):
    _plot_collection = az.plot_dist(
        idata,
        var_names=var_names,
        ci_prob=None,
        point_estimate=None,
    )

    for _var in var_names:
        _axis = _plot_collection.get_viz("plot", _var)
        _posterior_values = idata["posterior"].dataset[_var].values.ravel()
        _lower, _upper = np.quantile(_posterior_values, [0.005, 0.995])

        _axis.axvline(
            np.mean(_posterior_values),
            color="C0",
            linestyle="--",
            linewidth=2,
            label="posterior mean",
        )
        _axis.axvline(
            true_params[_var],
            color="red",
            linestyle="-",
            linewidth=2,
            label="true value",
        )
        _axis.set_xlim(_lower, _upper)

    _plot_collection
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Takeaway

    HSSM's analytical Poisson race likelihood can recover the generating
    accumulator rates, thresholds, and non-decision time in this synthetic
    example. Model-level prior presets are relevant when those parameters are
    regression targets; explicit parameter priors remain the appropriate tool
    for changing this intercept-only model.
    """)
    return


if __name__ == "__main__":
    app.run()
