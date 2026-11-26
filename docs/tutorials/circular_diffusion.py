"""Fit and check a circular diffusion model for continuous direction reports.

From a development checkout that contains native circular plotting, install the
matching ssm-simulators build (see the notebook), then regenerate the static page:

    python -m marimo export ipynb --no-sandbox docs/tutorials/circular_diffusion.py \
        --include-outputs --output docs/tutorials/circular_diffusion.ipynb --force
"""

# ruff: noqa: B018, E501, PLR1711  (notebook display expressions, prose, and bare returns)
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import warnings

    import arviz as az
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    import hssm

    # Keep backend notices out of the static page without hiding sampler warnings.
    for _name in ("hssm", "pymc"):
        logging.getLogger(_name).setLevel(logging.WARNING)
    warnings.filterwarnings(
        "ignore", message=r"'Model\..*' is deprecated", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore",
        message="Numba will use object mode",
        module=r"pytensor\.link\.numba\.",
    )
    hssm.set_floatX("float64")
    return az, hssm, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Fit response times and reported directions

    Suppose a participant points to the direction of a moving dot. A trial gives
    us an RT and an angle, not a correct/incorrect choice. In the circular diffusion
    model, evidence starts at the center of a disk. The location where it first
    hits the circle gives the angle; the time taken plus nondecision time gives RT.

    **Development example.** The native model and its two-panel predictive plot
    require the HSSM checkout containing this example. The circular sampler
    has merged into ssm-simulators but is not in the published 0.16.0 wheel.
    With the development HSSM checkout installed, install this tested revision
    from ssm-simulators main (no JEAM required):

    ```sh
    python -m pip install "ssm-simulators @ git+https://github.com/lnccbrown/ssm-simulators.git@b7578d085882d885afdaae2333e2ad346d8c2640"
    ```

    ## Simulate one participant

    Use the target as the zero direction. A positive `v_x` points toward the
    target; a positive `v_y` pulls the mean report counterclockwise. These
    are the two components of **one** drift vector. We give it a small sideways
    component here. RT is in seconds; the `response` column is an angle in radians
    on [−π, π). HSSM reads that circular domain from the simulator's model config;
    you don't need an extra column or a plotting option to declare it.
    """)
    return


@app.cell
def _(hssm, mo):
    truth = {"v_x": 1.5, "v_y": 0.5, "a": 1.0, "t": 0.2}
    data = hssm.simulate_data(
        "circular_diffusion", theta=truth, size=250, random_state=42
    )
    mo.Html(data.head().to_html(index=False, float_format="{:.3f}".format))
    return data, truth


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Estimate all four parameters

    We pass only the observations to the model. HSSM estimates the two drift
    components, the decision radius `a`, and nondecision time `t` together.
    Unit diffusion noise fixes the scale, and `p_outlier=None` turns off the
    lapse mixture. Drift is relative to our chosen target axis; its components
    are not separate participant groups or regression slopes.
    """)
    return


@app.cell
def _(data, hssm):
    model = hssm.HSSM(data=data, model="circular_diffusion", p_outlier=None)
    fit = model.sample(
        sampler="pymc",
        chains=4,
        cores=1,
        tune=600,
        draws=600,
        init="adapt_diag",
        target_accept=0.9,
        random_seed=43,
        progressbar=False,
    )
    return fit, model


@app.cell
def _(az, fit, mo, truth):
    _summary = az.summary(fit, var_names=list(truth), ci_kind="eti", ci_prob=0.95)[
        ["mean", "eti95_lb", "eti95_ub", "ess_bulk", "r_hat"]
    ].rename(columns={"eti95_lb": "2.5%", "eti95_ub": "97.5%"})
    _summary.insert(0, "generating", [truth[_name] for _name in _summary.index])
    mo.Html(_summary.to_html())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check both observed outcomes

    Simulate new RTs and angles from the fitted model, then use its default
    predictive plot. The left panel compares **RT densities**. On the right,
    the radius is the **fraction of reports in an angular bin**: a radius of
    10% means one report in ten fell in that sector. Each replicate is
    normalized separately. The shaded intervals are pointwise across replicated
    datasets, not intervals for the drift direction.
    """)
    return


@app.cell
def _(model, np, plt):
    model.sample_posterior_predictive(draws=np.arange(0, 600, 30), random_seed=44)
    rt_ax, _angle_ax = model.plot_predictive()
    _figure = rt_ax.figure
    plt.close(_figure)
    _figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    In the fitted target frame, both drift components are positive: the drift
    points toward the target but also to its counterclockwise side. This is
    a property of the simulated participant, not a claim about human bias.
    Look for shifts in the center or tails of the RT distribution and for
    sectors the model consistently over- or under-predicts. Agreement in these
    two panels checks the separate distributions of speed and direction; it does
    not by itself test whether slower individual reports have larger errors.
    """)
    return


if __name__ == "__main__":
    app.run()
