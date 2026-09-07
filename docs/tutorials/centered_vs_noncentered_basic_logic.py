# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "bambi==0.20.0",
#     "graphviz==0.21",
#     "hssm @ git+https://github.com/lnccbrown/HSSM.git@b6a6bdcf68ecd7cf71ffdef5cdda4fb05e8bfaad",
#     "marimo==0.24.0",
#     "matplotlib==3.11.1",
#     "numpy==2.4.6",
#     "pandas==3.0.5",
#     "pymc==6.3.1",
# ]
# ///

"""Explain centered and non-centered group effects in current HSSM.

This construction-only marimo tutorial separates the mathematical
reparameterization from Bambi's current implementation, demonstrates HSSM's
pre-build compatibility guard, and compares the model graphs for valid and
problematic group-location layouts. No sampling is required.

Run the pinned standalone environment locally or in Molab::

    uvx marimo edit --sandbox \
        docs/tutorials/centered_vs_noncentered_basic_logic.py

To exercise an active HSSM checkout instead, ignore the inline environment::

    uv run --group notebook --group docs marimo edit --no-sandbox \
        docs/tutorials/centered_vs_noncentered_basic_logic.py
    uv run --group notebook --group docs marimo check --strict \
        docs/tutorials/centered_vs_noncentered_basic_logic.py
    uv run --group notebook --group docs marimo export html --no-sandbox \
        docs/tutorials/centered_vs_noncentered_basic_logic.py \
        --output /tmp/centered-vs-noncentered.html --force
    uv run --group notebook --group docs marimo export ipynb --no-sandbox \
        docs/tutorials/centered_vs_noncentered_basic_logic.py \
        --output docs/tutorials/centered_vs_noncentered_basic_logic.ipynb \
        --include-outputs --force
    uv run ruff format \
        docs/tutorials/centered_vs_noncentered_basic_logic.ipynb
"""

# ruff: noqa: B018, D401, E501, PLR1711  (generated marimo notebook: prose, cell display expressions, and bare returns)
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import os
    import warnings
    from tempfile import gettempdir

    # This notebook only constructs models. Molab can expose a CUDA plugin even
    # when no usable GPU is present, so select CPU before importing HSSM/JAX.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ.setdefault(
        "MPLCONFIGDIR", f"{gettempdir()}/hssm-parameterization-matplotlib"
    )

    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    import bambi as bmb
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm
    from pytensor.graph.traversal import ancestors

    import hssm

    logging.getLogger("hssm").setLevel(logging.WARNING)
    hssm.set_floatX("float64")
    pd.set_option("display.max_colwidth", 100)
    return ancestors, bmb, hssm, logging, mo, np, pd, pm


@app.cell
def _(bmb, hssm, mo):
    mo.md(f"""
    # Centered vs. non-centered parameterizations

    Centering is a choice about **how a hierarchical effect is represented for
    computation**. It should not change the statistical model. That distinction
    becomes important when a group-specific prior has its own population mean.

    This tutorial uses **HSSM {hssm.__version__}** and **Bambi {bmb.__version__}**.
    It constructs models and inspects their PyMC graphs; no MCMC is run.

    By the end, you should be able to:

    1. distinguish the mathematical non-centered transformation from Bambi's
       current shortcut;
    2. recognize a valid zero-mean group deviation in either parameterization;
    3. understand why HSSM rejects an explicit non-centered free group mean; and
    4. choose exactly one owner for each population location.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Same distribution, different coordinates

    Let $u_g$ be a coefficient for group $g$, with population location
    $\mu$ and group scale $\sigma$.

    In the **centered** parameterization we sample the coefficient directly:

    $$
    u_g \sim \mathcal{N}(\mu, \sigma).
    $$

    A mathematically equivalent **non-centered** parameterization samples a
    standard-normal coordinate and transforms it:

    $$
    z_g \sim \mathcal{N}(0, 1),
    \qquad
    u_g = \mu + \sigma z_g.
    $$

    These equations define the same prior distribution for $u_g$. Which
    coordinates sample better depends on the amount of information in the data
    and the posterior geometry; neither form is universally superior.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    > **The current Bambi boundary**

    For a non-centered group term, Bambi currently constructs

    $$
    u_g = \sigma z_g,
    $$

    rather than $\mu + \sigma z_g$. This shortcut is faithful when the group
    term is a zero-mean deviation: `mu` is absent or fixed entirely to zero. A
    free or nonzero `mu` would be dropped.

    HSSM validates explicit group priors before asking Bambi to build the PyMC
    model. If that shortcut would discard part of the requested prior, HSSM
    raises a `ValueError` instead of silently changing the model.
    """)
    return


@app.cell
def _(ancestors, bmb, hssm, logging, np, pd):
    tutorial_data = pd.DataFrame(
        {
            "rt": 0.38 + 0.015 * np.arange(16),
            "response": np.where(np.arange(16) % 2, 1, -1),
            "theta": np.linspace(-1.0, 1.0, 16),
            "participant_id": np.repeat(np.arange(4), 4),
        }
    )
    raw_bambi_data = tutorial_data.loc[:, ["participant_id"]].assign(
        y=np.array(
            [
                -0.20,
                0.05,
                0.10,
                -0.05,
                0.30,
                0.35,
                0.20,
                0.40,
                -0.35,
                -0.20,
                -0.10,
                -0.25,
                0.15,
                0.25,
                0.05,
                0.30,
            ]
        )
    )

    def zero_mean_group_prior(*, noncentered=None):
        """Return a fresh zero-mean hierarchical Normal group prior."""
        return hssm.Prior(
            "Normal",
            mu=0.0,
            sigma=hssm.Prior("HalfNormal", sigma=0.5),
            noncentered=noncentered,
        )

    def free_location_group_prior(*, noncentered=None):
        """Return a fresh hierarchical Normal with a free population mean."""
        return hssm.Prior(
            "Normal",
            mu=hssm.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=hssm.Prior("HalfNormal", sigma=0.5),
            noncentered=noncentered,
        )

    def matched_include(group_prior):
        """Place a common intercept and matching group intercept in one spec."""
        return [
            {
                "name": "v",
                "formula": "v ~ 1 + (1|participant_id)",
                "prior": {
                    "Intercept": hssm.Prior("Normal", mu=0.0, sigma=0.5),
                    "1|participant_id": group_prior,
                },
            }
        ]

    def hssm_model(*, formula, priors, noncentered=True):
        """Build a tiny analytical DDM without sampling or init-value work."""
        return hssm.HSSM(
            data=tutorial_data,
            model="ddm",
            loglik_kind="analytical",
            include=[{"name": "v", "formula": formula, "prior": priors}],
            p_outlier=0.0,
            prior_settings=None,
            noncentered=noncentered,
            process_initvals=False,
            initval_jitter=0.0,
        )

    def disconnected_free_rvs(pymc_model):
        """Return free-RV names that are not ancestors of observed variables."""
        connected = {
            id(variable)
            for observed_rv in pymc_model.observed_RVs
            for variable in ancestors([observed_rv])
        }
        return sorted(rv.name for rv in pymc_model.free_RVs if id(rv) not in connected)

    def capture_hssm_build(builder):
        """Build a model while collecting HSSM warning messages."""
        logger = logging.getLogger("hssm")
        messages = []

        class _MessageHandler(logging.Handler):
            def emit(self, record):
                messages.append(record.getMessage())

        handler = _MessageHandler(level=logging.WARNING)
        previous_handlers = list(logger.handlers)
        previous_level = logger.level
        previous_propagate = logger.propagate
        logger.handlers = [handler]
        logger.setLevel(logging.WARNING)
        logger.propagate = False
        try:
            result = builder()
        finally:
            logger.handlers = previous_handlers
            logger.setLevel(previous_level)
            logger.propagate = previous_propagate
        return result, tuple(messages)

    # Assert the factories return fresh prior trees. HSSM/Bambi attach names
    # while preparing priors, so tutorial cases must not share mutable objects.
    assert zero_mean_group_prior() is not zero_mean_group_prior()
    assert free_location_group_prior() is not free_location_group_prior()
    assert isinstance(free_location_group_prior().args["mu"], bmb.Prior)
    return (
        capture_hssm_build,
        disconnected_free_rvs,
        free_location_group_prior,
        hssm_model,
        matched_include,
        raw_bambi_data,
        tutorial_data,
        zero_mean_group_prior,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. A valid zero-mean group deviation

    Consider

    ```python
    v ~ 1 + (1 | participant_id)
    ```

    The common `Intercept` owns the population location. The participant term
    is a deviation around that location, so its prior has `mu=0` and a
    hierarchical `sigma`. This statistical model can be represented faithfully
    in either parameterization.
    """)
    return


@app.cell
def _(
    capture_hssm_build,
    disconnected_free_rvs,
    hssm,
    matched_include,
    pd,
    tutorial_data,
    zero_mean_group_prior,
):
    matched_centered_model, matched_centered_messages = capture_hssm_build(
        lambda: hssm.HSSM(
            data=tutorial_data,
            model="ddm",
            loglik_kind="analytical",
            include=matched_include(zero_mean_group_prior()),
            p_outlier=0.0,
            prior_settings=None,
            noncentered=False,
            process_initvals=False,
            initval_jitter=0.0,
        )
    )
    matched_noncentered_model, matched_noncentered_messages = capture_hssm_build(
        lambda: hssm.HSSM(
            data=tutorial_data,
            model="ddm",
            loglik_kind="analytical",
            include=matched_include(zero_mean_group_prior()),
            p_outlier=0.0,
            prior_settings=None,
            noncentered=True,
            process_initvals=False,
            initval_jitter=0.0,
        )
    )

    _centered_names = {rv.name for rv in matched_centered_model.pymc_model.free_RVs}
    _noncentered_names = {
        rv.name for rv in matched_noncentered_model.pymc_model.free_RVs
    }
    assert matched_centered_messages == ()
    assert matched_noncentered_messages == ()
    assert "v_1|participant_id" in _centered_names
    assert "v_1|participant_id_offset" not in _centered_names
    assert "v_1|participant_id_offset" in _noncentered_names
    assert "v_1|participant_id" not in _noncentered_names
    assert "v_1|participant_id_mu" not in _centered_names | _noncentered_names
    assert disconnected_free_rvs(matched_centered_model.pymc_model) == []
    assert disconnected_free_rvs(matched_noncentered_model.pymc_model) == []

    matched_parameterization_table = pd.DataFrame(
        [
            {
                "effective parameterization": "centered",
                "sampled group coordinate": "v_1|participant_id",
                "group coefficient in graph": "free RV",
                "disconnected free RVs": "none",
            },
            {
                "effective parameterization": "non-centered",
                "sampled group coordinate": "v_1|participant_id_offset",
                "group coefficient in graph": "deterministic",
                "disconnected free RVs": "none",
            },
        ]
    )
    matched_parameterization_table
    return (
        matched_centered_model,
        matched_noncentered_model,
        matched_parameterization_table,
    )


@app.cell
def _(matched_centered_model, pm):
    matched_centered_graph = pm.model_to_graphviz(matched_centered_model.pymc_model)
    matched_centered_graph
    return (matched_centered_graph,)


@app.cell
def _(mo):
    mo.md("""
    In the centered graph, `v_1|participant_id` is sampled directly. Its scale
    hyperprior points to the group coefficients, and those coefficients point
    to the trial-wise drift rate.
    """)
    return


@app.cell
def _(matched_noncentered_model, pm):
    matched_noncentered_graph = pm.model_to_graphviz(
        matched_noncentered_model.pymc_model
    )
    matched_noncentered_graph
    return (matched_noncentered_graph,)


@app.cell
def _(mo):
    mo.md("""
    In the non-centered graph, Bambi samples a standard-normal `offset` and
    combines it with `sigma`. The resulting `v_1|participant_id` is a
    deterministic node. Because the intended group mean is exactly zero, this
    graph represents the same prior as the centered graph above.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Why HSSM needs a pre-build guard

    To isolate the underlying mechanism, the next example uses Bambi directly
    with a tiny Gaussian response. The requested group prior has a free
    population location:

    ```python
    Normal(
        mu=Normal(0, 0.5),
        sigma=HalfNormal(0.5),
    )
    ```

    This is a legitimate centered hierarchy. Under Bambi's current
    non-centered shortcut, however, the `mu` prior is created and then omitted
    from `offset * sigma`.
    """)
    return


@app.cell
def _(bmb, disconnected_free_rvs, pd, raw_bambi_data):
    raw_bambi_group_prior = bmb.Prior(
        "Normal",
        mu=bmb.Prior("Normal", mu=0.0, sigma=0.5),
        sigma=bmb.Prior("HalfNormal", sigma=0.5),
    )
    raw_bambi_model = bmb.Model(
        "y ~ 1 + (1|participant_id)",
        raw_bambi_data,
        family="gaussian",
        priors={"1|participant_id": raw_bambi_group_prior},
        noncentered=True,
    )
    raw_bambi_model.build()
    raw_bambi_pymc_model = raw_bambi_model.backend.model
    raw_bambi_orphans = disconnected_free_rvs(raw_bambi_pymc_model)

    _free_names = [rv.name for rv in raw_bambi_pymc_model.free_RVs]
    assert "1|participant_id_mu" in _free_names
    assert "1|participant_id_offset" in _free_names
    assert raw_bambi_orphans == ["1|participant_id_mu"]

    raw_bambi_summary = pd.DataFrame(
        {
            "quantity": ["sampled group coordinates", "disconnected free RVs"],
            "value": [", ".join(_free_names), ", ".join(raw_bambi_orphans)],
        }
    )
    raw_bambi_summary
    return raw_bambi_model, raw_bambi_orphans, raw_bambi_pymc_model, raw_bambi_summary


@app.cell
def _(pm, raw_bambi_pymc_model):
    raw_bambi_orphan_graph = pm.model_to_graphviz(raw_bambi_pymc_model)
    raw_bambi_orphan_graph
    return (raw_bambi_orphan_graph,)


@app.cell
def _(mo):
    mo.md("""
    > The floating `1|participant_id_mu` node has no path to the observed
    > response. Sampling it would consume computation without changing the
    > likelihood. This graph documents the **raw Bambi behavior that HSSM
    > prevents**; it is not a graph that current HSSM will construct from the
    > same explicit prior.
    """)
    return


@app.cell
def _(free_location_group_prior, hssm, matched_include, mo, tutorial_data):
    try:
        hssm.HSSM(
            data=tutorial_data,
            model="ddm",
            loglik_kind="analytical",
            include=matched_include(free_location_group_prior()),
            p_outlier=0.0,
            prior_settings=None,
            noncentered=True,
            process_initvals=False,
            initval_jitter=0.0,
        )
    except ValueError as exc:
        hssm_preflight_error = str(exc)
    else:
        raise AssertionError("HSSM did not reject the incompatible group prior")

    assert "cannot be represented faithfully by bambi" in hssm_preflight_error
    assert "1|participant_id" in hssm_preflight_error
    assert "omits from `offset * sigma`" in hssm_preflight_error
    assert "noncentered=False" in hssm_preflight_error
    mo.md(f"""
    **Current HSSM result**

    ```text
    {hssm_preflight_error}
    ```
    """)
    return (hssm_preflight_error,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Centering retains the mean—but location ownership still matters

    If we center the explicit hierarchy, Bambi uses its `mu`. With both a
    common intercept and a free group mean, the predictor for participant $g$
    contains

    $$
    \eta_g = \beta_0 + u_g,
    \qquad
    u_g \sim \mathcal{N}(\mu_u, \sigma_u).
    $$

    The likelihood sees $\beta_0 + \mu_u$, not the two locations separately.
    Shifting one up and the other down leaves the predictor unchanged. The PyMC
    graph is fully connected, but the likelihood has a ridge along that shift
    direction.
    """)
    return


@app.cell
def _(
    capture_hssm_build,
    disconnected_free_rvs,
    free_location_group_prior,
    hssm,
    matched_include,
    mo,
    tutorial_data,
):
    centered_ridge_model, centered_ridge_messages = capture_hssm_build(
        lambda: hssm.HSSM(
            data=tutorial_data,
            model="ddm",
            loglik_kind="analytical",
            include=matched_include(free_location_group_prior()),
            p_outlier=0.0,
            prior_settings=None,
            noncentered=False,
            process_initvals=False,
            initval_jitter=0.0,
        )
    )
    centered_ridge_warning = next(
        message
        for message in centered_ridge_messages
        if "non-identifiable individually" in message
    )

    assert "posterior will have a ridge" in centered_ridge_warning
    assert "common 'Intercept' effect" in centered_ridge_warning
    assert disconnected_free_rvs(centered_ridge_model.pymc_model) == []
    assert "v_1|participant_id_mu" in {
        rv.name for rv in centered_ridge_model.pymc_model.free_RVs
    }
    mo.md(f"""
    **HSSM's centered location warning**

    ```text
    {centered_ridge_warning}
    ```

    The disconnected-node check returns `[]`: this is an identifiability
    problem in the likelihood, not an orphan-node problem.
    """)
    return centered_ridge_model, centered_ridge_warning


@app.cell
def _(centered_ridge_model, pm):
    centered_ridge_graph = pm.model_to_graphviz(centered_ridge_model.pymc_model)
    centered_ridge_graph
    return (centered_ridge_graph,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. A valid group-owned location

    A free group mean is appropriate when the group term is the unique owner of
    that population location. Remove the matching common intercept:

    ```python
    v ~ 0 + (1 | participant_id)
    ```

    The outer group prior below carries `noncentered=False`. This per-prior
    setting overrides the model-level `noncentered=True`, retaining the
    requested $\mu_u$ without changing the parameterization of unrelated
    components.
    """)
    return


@app.cell
def _(
    capture_hssm_build,
    disconnected_free_rvs,
    free_location_group_prior,
    hssm_model,
    pd,
):
    unique_owner_model, unique_owner_messages = capture_hssm_build(
        lambda: hssm_model(
            formula="v ~ 0 + (1|participant_id)",
            priors={"1|participant_id": free_location_group_prior(noncentered=False)},
            noncentered=True,
        )
    )
    _free_names = {rv.name for rv in unique_owner_model.pymc_model.free_RVs}
    assert unique_owner_messages == ()
    assert "v_Intercept" not in unique_owner_model.pymc_model.named_vars
    assert "v_1|participant_id_mu" in _free_names
    assert "v_1|participant_id" in _free_names
    assert "v_1|participant_id_offset" not in _free_names
    assert disconnected_free_rvs(unique_owner_model.pymc_model) == []

    unique_owner_summary = pd.DataFrame(
        [
            {
                "model default": "non-centered",
                "group-prior override": "centered",
                "population-location owner": "1|participant_id mu",
                "disconnected free RVs": "none",
            }
        ]
    )
    unique_owner_summary
    return unique_owner_model, unique_owner_summary


@app.cell
def _(pm, unique_owner_model):
    unique_owner_graph = pm.model_to_graphviz(unique_owner_model.pymc_model)
    unique_owner_graph
    return (unique_owner_graph,)


@app.cell
def _(mo):
    mo.md("""
    The group `mu` now has a path through the direct group coefficients to the
    likelihood, and there is no competing `v_Intercept`. HSSM's generated safe
    priors use this centered fallback automatically when one unmatched group
    term is the unique population-location owner.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Choose one owner for every population location

    The rule is structural and applies to intercepts and slopes alike:

    - `x + (0 + x | participant_id)`: common `x` owns the population slope;
      the participant coefficients are zero-mean deviations.
    - `(0 + x | participant_id)` with no common `x`: that single group term may
      own the population slope, so a free location must be effectively centered.
    - `(0 + x | participant_id) + (0 + x | item_id)` with no common `x`: two
      group means compete for the same location. Add common `x` and make both
      group terms zero-mean, or deliberately choose exactly one owner.

    An inverse link does not change this logic. Common and group coefficients
    are combined on the linear-predictor scale before the inverse link is
    applied.
    """)
    return


@app.cell
def _(pd):
    parameterization_rules = pd.DataFrame(
        [
            {
                "formula structure": "exact common/group match",
                "location owner": "common term",
                "group mu": "fixed at zero",
                "HSSM result": "centered or non-centered is valid",
            },
            {
                "formula structure": "one unmatched group term",
                "location owner": "group distribution",
                "group mu": "may be free",
                "HSSM result": "use effective centering",
            },
            {
                "formula structure": "repeated unmatched group expression",
                "location owner": "ambiguous until specified",
                "group mu": "do not free every mean",
                "HSSM result": "safe generation rejects; explicit ridges warn",
            },
            {
                "formula structure": "explicit incompatible NC prior",
                "location owner": "would be discarded by Bambi",
                "group mu": "free or nonzero",
                "HSSM result": "pre-build ValueError",
            },
        ]
    )
    parameterization_rules
    return (parameterization_rules,)


@app.cell
def _(mo):
    mo.md("""
    ## 7. Where to continue

    - [Specify hierarchical group priors](https://lnccbrown.github.io/HSSM/how_to/specify_group_priors/)
      gives the complete explicit-prior compatibility contract.
    - [Link functions and safe priors](https://lnccbrown.github.io/HSSM/tutorials/link_functions/)
      explains why population locations live on the linear-predictor scale.
    - [Choosing a parameterization per parameter](https://lnccbrown.github.io/HSSM/tutorials/parameterization_per_parameter/)
      shows model, component, and per-prior override precedence.
    - Betancourt's [hierarchical modeling case study](https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html)
      develops the posterior geometry behind centered and non-centered choices.

    The practical takeaway is simple: parameterization may change coordinates,
    but it must not change who owns a population location. HSSM's preflight
    checks enforce that boundary before Bambi constructs a different graph.
    """)
    return


if __name__ == "__main__":
    app.run()
