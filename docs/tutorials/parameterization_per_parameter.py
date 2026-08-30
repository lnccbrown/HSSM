# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "bambi==0.20.0",
#     "graphviz==0.21",
#     "hssm @ git+https://github.com/lnccbrown/HSSM.git@b6a6bdcf68ecd7cf71ffdef5cdda4fb05e8bfaad",
#     "marimo==0.24.0",
#     "numpy==2.4.6",
#     "pandas==3.0.5",
#     "pymc==6.3.1",
# ]
# ///

"""Choose centered or non-centered group effects at each HSSM control level.

This construction-only marimo tutorial is the source of truth for the rendered
Jupyter artifact published by MkDocs. It uses tiny synthetic data, inspects the
PyMC graphs, and exercises HSSM's current group-prior preflight without sampling.

Run the pinned standalone environment locally or in Molab::

    uvx marimo edit --sandbox docs/tutorials/parameterization_per_parameter.py

To exercise an active HSSM checkout instead, ignore the inline environment::

    uv run --group notebook --group docs marimo edit --no-sandbox \
        docs/tutorials/parameterization_per_parameter.py
    uv run --group notebook --group docs marimo check --strict \
        docs/tutorials/parameterization_per_parameter.py
    uv run --group notebook --group docs marimo export html --no-sandbox \
        docs/tutorials/parameterization_per_parameter.py \
        --output /tmp/parameterization-per-parameter.html --force
    uv run --group notebook --group docs marimo export ipynb --no-sandbox \
        docs/tutorials/parameterization_per_parameter.py \
        --output docs/tutorials/parameterization_per_parameter.ipynb \
        --include-outputs --force
    uv run ruff format docs/tutorials/parameterization_per_parameter.ipynb

Graph cells require the Graphviz ``dot`` executable (for example,
``brew install graphviz`` on macOS). No sampling is performed.
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
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO
    from tempfile import gettempdir

    # This notebook only constructs graphs. Molab and some development hosts can
    # expose an unusable CUDA plugin, so prevent JAX from probing it.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    os.environ.setdefault(
        "MPLCONFIGDIR", f"{gettempdir()}/hssm-per-parameter-matplotlib"
    )
    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

    import bambi as bmb
    import jax
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm

    # HSSM reports backend configuration and registry details during import and
    # setup. Suppress that incidental output; later helpers capture HSSM's
    # parameterization warnings explicitly and display them in the relevant cell.
    with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
        import hssm
        from hssm.param.parameterization_check import find_disconnected_free_rvs

        hssm.set_floatX("float64")

    assert jax.default_backend() == "cpu"
    pd.set_option("display.max_colwidth", 100)
    return (
        StringIO,
        bmb,
        find_disconnected_free_rvs,
        hssm,
        jax,
        logging,
        mo,
        np,
        pd,
        pm,
        redirect_stderr,
        redirect_stdout,
    )


@app.cell(hide_code=True)
def _(bmb, hssm, jax, mo):
    mo.md(f"""
    # Choosing a parameterization per parameter

    Hierarchical models often sample better when different parameters—or even
    different group terms—use different parameterizations. This tutorial shows
    how HSSM resolves those choices and how to verify the resulting PyMC graph.

    By the end, you will be able to:

    1. choose a model-wide or component-specific default;
    2. override one explicit group prior safely;
    3. recognize HSSM's pre-build errors and location-ridge warnings; and
    4. decide which formula term owns each population location.

    Everything below is structural: the models are built but never sampled.

    **Environment:** HSSM `{hssm.__version__}`, Bambi `{bmb.__version__}`,
    JAX `{jax.__version__}` on `{jax.default_backend()}`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Centering is both geometry and model structure

    A textbook hierarchical Normal can be written in centered form,

    \[
    b_g \sim \mathcal N(\mu, \sigma),
    \]

    or in the mathematically equivalent non-centered form,

    \[
    z_g \sim \mathcal N(0,1), \qquad b_g = \mu + \sigma z_g.
    \]

    The likelihood can be identical while the posterior geometry—and therefore
    sampling efficiency—changes. Non-centering is often helpful for weakly
    informed groups; centering can be better for strongly informed groups.

    There is one important implementation boundary. Current Bambi constructs a
    non-centered group term as

    \[
    b_g = \sigma z_g,
    \]

    so this route faithfully represents a built-in Normal group prior only when
    its location is absent or fixed entirely to zero and `sigma` is hierarchical.
    HSSM checks explicit priors before asking Bambi to build the model. A free or
    nonzero `mu` under effective non-centering now raises an actionable error; it
    is not silently accepted and then discarded.

    The [hierarchical group-prior guide](https://lnccbrown.github.io/HSSM/how_to/specify_group_priors/)
    gives the complete compatibility table. For the scale on which these effects
    combine, see [Link functions and safe priors](https://lnccbrown.github.io/HSSM/tutorials/link_functions/).
    """)
    return


@app.cell
def _(np, pd):
    _trial = np.arange(24)
    tutorial_data = pd.DataFrame(
        {
            "rt": 0.42 + 0.01 * _trial,
            "response": np.where(_trial % 2, 1, -1),
            "theta": np.tile([-1.0, 0.0, 1.0], 8),
            "participant_id": np.repeat(np.arange(4), 6),
            "conf": np.tile(["low", "high"], 12),
        }
    )
    tutorial_data.head(8)
    return (tutorial_data,)


@app.cell
def _(
    StringIO,
    find_disconnected_free_rvs,
    hssm,
    logging,
    pm,
    redirect_stderr,
    redirect_stdout,
    tutorial_data,
):
    _base_model_kwargs = {
        "data": tutorial_data,
        "model": "ddm",
        "loglik_kind": "analytical",
        "p_outlier": 0.0,
        "prior_settings": "safe",
        "process_initvals": False,
        "initval_jitter": 0.0,
        "z": 0.5,
        "t": 0.2,
    }

    def build_model(include, **kwargs):
        """Build quietly while returning only HSSM warning messages."""
        _log_stream = StringIO()
        _handler = logging.StreamHandler(_log_stream)
        _handler.setFormatter(logging.Formatter("%(message)s"))
        _logger = logging.getLogger("hssm")
        _old_handlers = list(_logger.handlers)
        _old_level = _logger.level
        _old_propagate = _logger.propagate
        _logger.handlers = [_handler]
        _logger.setLevel(logging.WARNING)
        _logger.propagate = False
        try:
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                _model = hssm.HSSM(
                    **_base_model_kwargs,
                    include=include,
                    **kwargs,
                )
        finally:
            _logger.handlers = _old_handlers
            _logger.setLevel(_old_level)
            _logger.propagate = _old_propagate
        _messages = tuple(
            _line.strip() for _line in _log_stream.getvalue().splitlines() if _line
        )
        return _model, _messages

    def expect_model_error(include, **kwargs):
        """Return the expected pre-build ValueError as stable tutorial evidence."""
        try:
            build_model(include, **kwargs)
        except ValueError as _error:
            return str(_error)
        raise AssertionError("HSSM unexpectedly built an incompatible model")

    def free_rv_names(model):
        """Return exact free-RV names; do not infer component names by splitting."""
        return {variable.name for variable in model.pymc_model.free_RVs}

    def group_term_structure(model, parameter, term):
        """Summarize one exact group key in the built PyMC graph."""
        _prefix = f"{parameter}_{term}"
        _free = free_rv_names(model)
        _prior = model.params[parameter].prior[term]
        return {
            "parameter": parameter,
            "group term": term,
            "prior override": getattr(_prior, "noncentered", None),
            "effective form": (
                "non-centered" if f"{_prefix}_offset" in _free else "centered"
            ),
            "direct group RV": _prefix in _free,
            "offset RV": f"{_prefix}_offset" in _free,
            "free mu RV": f"{_prefix}_mu" in _free,
            "free sigma RV": f"{_prefix}_sigma" in _free,
            "disconnected RVs": ", ".join(find_disconnected_free_rvs(model.pymc_model))
            or "none",
        }

    def assert_connected(model):
        """Make every successful example double as a graph regression."""
        assert find_disconnected_free_rvs(model.pymc_model) == []

    def model_graph(model):
        """Render a compact construction-only PyMC graph."""
        return pm.model_to_graphviz(
            model.pymc_model,
            graph_attr={"bgcolor": "white", "rankdir": "LR"},
        )

    return (
        assert_connected,
        build_model,
        expect_model_error,
        free_rv_names,
        group_term_structure,
        model_graph,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Three levels of control

    HSSM passes a scalar `noncentered=True` or `False` to Bambi as the default
    for every group term. A dictionary selects defaults by HSSM parameter name.
    Missing dictionary keys fall back to `True`, not to the value of another
    component. Finally, `noncentered` on an explicit prior wins for that one term.

    The formulas below contain matching common and group intercepts. The common
    `Intercept` owns the population location, so each generated group intercept is
    a mean-zero deviation and either parameterization is faithful.
    """)
    return


@app.cell
def _(assert_connected, build_model):
    hierarchical_specs = [
        {"name": "v", "formula": "v ~ 1 + (1 | participant_id)"},
        {"name": "a", "formula": "a ~ 1 + (1 | participant_id)"},
    ]

    _scalar_noncentered_model, _scalar_nc_messages = build_model(
        hierarchical_specs,
        noncentered=True,
    )
    _scalar_centered_model, _scalar_c_messages = build_model(
        hierarchical_specs,
        noncentered=False,
    )
    _component_dict_model, _component_messages = build_model(
        hierarchical_specs,
        noncentered={"v": False},
    )

    parameterization_models = {
        "scalar True": _scalar_noncentered_model,
        "scalar False": _scalar_centered_model,
        "dict: v=False; a omitted": _component_dict_model,
    }
    assert not (_scalar_nc_messages or _scalar_c_messages or _component_messages)
    for _model in parameterization_models.values():
        assert_connected(_model)
    return hierarchical_specs, parameterization_models


@app.cell
def _(group_term_structure, parameterization_models, pd):
    _rows = []
    for _setting, _model in parameterization_models.items():
        for _parameter in ("v", "a"):
            _row = group_term_structure(_model, _parameter, "1|participant_id")
            _row = {"model setting": _setting, **_row}
            _rows.append(_row)

    parameterization_table = pd.DataFrame(_rows)[
        [
            "model setting",
            "parameter",
            "effective form",
            "direct group RV",
            "offset RV",
            "free mu RV",
            "disconnected RVs",
        ]
    ]
    assert parameterization_table.loc[
        parameterization_table["model setting"] == "scalar True", "offset RV"
    ].all()
    assert parameterization_table.loc[
        parameterization_table["model setting"] == "scalar False", "direct group RV"
    ].all()
    _dict_rows = parameterization_table[
        parameterization_table["model setting"] == "dict: v=False; a omitted"
    ].set_index("parameter")
    assert _dict_rows.loc["v", "effective form"] == "centered"
    assert _dict_rows.loc["a", "effective form"] == "non-centered"
    parameterization_table
    return (parameterization_table,)


@app.cell
def _(model_graph, parameterization_models):
    mixed_parameterization_graph = model_graph(
        parameterization_models["dict: v=False; a omitted"]
    )
    mixed_parameterization_graph
    return (mixed_parameterization_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    In the dictionary case, `v_1|participant_id` is sampled directly because
    `v` is centered. The missing `a` key takes Bambi's default `True`, so
    `a_1|participant_id_offset` is sampled and multiplied by its scale. Neither
    graph contains a group `mu`: these are zero-mean deviations around their
    matching common intercepts.

    The graph below fixes the mixed dictionary case as the same static view in
    marimo, Molab, and the rendered documentation.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## A per-prior override wins

    The finest control lives on the group prior itself. Here the component
    dictionary requests centered `v`, while this one explicit prior requests
    non-centering. Its `mu=0` and hierarchical `sigma` satisfy Bambi's current
    non-centered contract.
    """)
    return


@app.cell
def _(assert_connected, build_model, free_rv_names, hssm):
    zero_mean_noncentered_prior = hssm.Prior(
        "Normal",
        mu=0.0,
        sigma=hssm.Prior("HalfNormal", sigma=0.5),
        noncentered=True,
    )
    _override_include = [
        {
            "name": "v",
            "formula": "v ~ 1 + (1 | participant_id)",
            "prior": {"1|participant_id": zero_mean_noncentered_prior},
        }
    ]
    override_model, override_messages = build_model(
        _override_include,
        a=1.5,
        noncentered={"v": False},
    )
    _override_free = free_rv_names(override_model)
    assert not override_messages
    assert "v_1|participant_id_offset" in _override_free
    assert "v_1|participant_id_mu" not in _override_free
    assert_connected(override_model)
    return override_model, zero_mean_noncentered_prior


@app.cell
def _(group_term_structure, override_model, pd):
    override_structure_table = pd.DataFrame(
        [group_term_structure(override_model, "v", "1|participant_id")]
    )
    override_structure_table
    return (override_structure_table,)


@app.cell
def _(model_graph, override_model):
    faithful_noncentered_graph = model_graph(override_model)
    faithful_noncentered_graph
    return (faithful_noncentered_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The graph contains the group scale and standard-normal offset, but no
    `v_1|participant_id_mu`. This is not an orphan-removal trick: zero is the
    intended location because the common `v_Intercept` owns the population mean.

    The complete resolution order is:
    """)
    return


@app.cell
def _(pd):
    precedence_table = pd.DataFrame(
        [
            {
                "priority": 1,
                "control": "per-prior noncentered",
                "scope": "one explicit group term",
                "rule": "wins when True or False",
            },
            {
                "priority": 2,
                "control": "model-level component dictionary",
                "scope": "all group terms for that HSSM parameter",
                "rule": "named key wins; missing key defaults to True",
            },
            {
                "priority": 3,
                "control": "model-level scalar",
                "scope": "all group terms",
                "rule": "True is the default; False requests centering",
            },
            {
                "priority": "safe-policy override",
                "control": "generated unique group-only prior",
                "scope": "only the generated location-owning term",
                "rule": "HSSM centers it to preserve its location",
            },
        ]
    )
    precedence_table
    return (precedence_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The last row is deliberately separate from user precedence. HSSM may adapt
    a prior that it generated itself, but an explicit user prior remains
    authoritative and is never rewritten.

    ## Incompatible explicit non-centered locations fail before build

    A free group mean and a fixed nonzero group mean would both be omitted from
    Bambi's `offset * sigma` construction. HSSM therefore rejects both before a
    PyMC model exists.
    """)
    return


@app.cell
def _(expect_model_error, hssm, pd):
    _free_mu_prior = hssm.Prior(
        "Normal",
        mu=hssm.Prior("Normal", mu=0.0, sigma=0.5),
        sigma=hssm.Prior("HalfNormal", sigma=0.5),
    )
    _nonzero_mu_prior = hssm.Prior(
        "Normal",
        mu=1.0,
        sigma=hssm.Prior("HalfNormal", sigma=0.5),
    )

    _free_mu_include = [
        {
            "name": "v",
            "formula": "v ~ 1 + (1 | participant_id)",
            "prior": {"1|participant_id": _free_mu_prior},
        }
    ]
    _nonzero_mu_include = [
        {
            "name": "v",
            "formula": "v ~ 1 + (1 | participant_id)",
            "prior": {"1|participant_id": _nonzero_mu_prior},
        }
    ]
    free_mu_error = expect_model_error(
        _free_mu_include,
        a=1.5,
        noncentered=True,
    )
    nonzero_mu_error = expect_model_error(
        _nonzero_mu_include,
        a=1.5,
        noncentered=True,
    )

    assert free_mu_error.startswith(
        "Explicit group-specific prior specification(s) cannot be represented"
    )
    assert "mu` hyperprior" in free_mu_error
    assert "disconnected node" in free_mu_error
    assert "not fixed entirely to zero" in nonzero_mu_error
    assert "silently ignored" in nonzero_mu_error

    incompatible_location_table = pd.DataFrame(
        [
            {
                "explicit group mu": "Normal hyperprior (free)",
                "effective setting": "non-centered",
                "HSSM result": "ValueError before Bambi/PyMC build",
                "reason": "Bambi would create mu but omit it from offset * sigma",
            },
            {
                "explicit group mu": "1.0 (fixed nonzero)",
                "effective setting": "non-centered",
                "HSSM result": "ValueError before Bambi/PyMC build",
                "reason": "Bambi would ignore the requested location",
            },
        ]
    )
    incompatible_location_table
    return free_mu_error, incompatible_location_table, nonzero_mu_error


@app.cell(hide_code=True)
def _(free_mu_error, mo):
    mo.md(f"""
    HSSM reports both the underlying limitation and term-specific repairs. The
    free-mean case says:

    ```text
    {free_mu_error}
    ```

    For a matched common/group expression, keep the common effect and use a
    zero-mean group deviation. If the group distribution should own a free
    population location, remove the matching common term and center that group
    prior intentionally.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## A unique group-only term owns its location

    Now `theta` appears only inside `(0 + theta | participant_id)`. The group
    distribution must estimate the population slope: fixing its mean to zero
    would change the scientific model. With `prior_settings="safe"`, HSSM keeps
    that generated location and centers just this term—even when the model-level
    request is non-centered.
    """)
    return


@app.cell
def _(assert_connected, build_model, free_rv_names):
    _group_only_include = [
        {
            "name": "v",
            "formula": "v ~ 1 + (0 + theta | participant_id)",
        }
    ]
    generated_owner_model, generated_owner_messages = build_model(
        _group_only_include,
        a=1.5,
        noncentered=True,
    )
    _owner_prior = generated_owner_model.params["v"].prior["theta|participant_id"]
    _owner_free = free_rv_names(generated_owner_model)
    assert _owner_prior.noncentered is False
    assert "v_theta|participant_id" in _owner_free
    assert "v_theta|participant_id_mu" in _owner_free
    assert "v_theta|participant_id_offset" not in _owner_free
    assert len(generated_owner_messages) == 1
    assert "generated location-bearing group-only term" in generated_owner_messages[0]
    assert "Explicit priors were not changed" in generated_owner_messages[0]
    assert_connected(generated_owner_model)
    return generated_owner_messages, generated_owner_model


@app.cell(hide_code=True)
def _(generated_owner_messages, mo):
    mo.md(f"""
    HSSM makes the generated fallback visible:

    ```text
    {generated_owner_messages[0]}
    ```

    In the graph, `v_theta|participant_id_mu` and the scale both feed the direct
    group coefficient. There is no offset and no disconnected node.
    """)
    return


@app.cell
def _(generated_owner_model, group_term_structure, pd):
    generated_owner_table = pd.DataFrame(
        [group_term_structure(generated_owner_model, "v", "theta|participant_id")]
    )
    generated_owner_table
    return (generated_owner_table,)


@app.cell
def _(generated_owner_model, model_graph):
    generated_owner_graph = model_graph(generated_owner_model)
    generated_owner_graph
    return (generated_owner_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Centered does not automatically mean identifiable

    Centering makes a free group `mu` part of the model, but the formula still
    needs exactly one population-location owner. If common `theta` and the mean
    of `theta|participant_id` are both free, the likelihood sees only their sum.
    HSSM can build this model faithfully, so it warns about the location ridge
    instead of rejecting the prior.
    """)
    return


@app.cell
def _(assert_connected, build_model, hssm):
    _matched_free_location = hssm.Prior(
        "Normal",
        mu=hssm.Prior("Normal", mu=0.0, sigma=0.5),
        sigma=hssm.Prior("HalfNormal", sigma=0.5),
    )
    _matched_ridge_include = [
        {
            "name": "v",
            "formula": "v ~ 1 + theta + (0 + theta | participant_id)",
            "prior": {"theta|participant_id": _matched_free_location},
        }
    ]
    matched_ridge_model, matched_ridge_messages = build_model(
        _matched_ridge_include,
        a=1.5,
        noncentered=False,
    )
    assert len(matched_ridge_messages) == 1
    assert "non-identifiable" in matched_ridge_messages[0]
    assert "common 'theta'" in matched_ridge_messages[0]
    assert "disconnected" not in matched_ridge_messages[0].lower()
    assert_connected(matched_ridge_model)
    return matched_ridge_messages, matched_ridge_model


@app.cell(hide_code=True)
def _(matched_ridge_messages, mo):
    mo.md(f"""
    ```text
    {matched_ridge_messages[0]}
    ```

    The graph is connected, but connectivity is not identifiability. Both
    `v_theta` and `v_theta|participant_id_mu` shift the same predictor. Keep
    common `theta` and set the group `mu=0`, or remove common `theta` and let the
    centered group distribution own the location.
    """)
    return


@app.cell
def _(group_term_structure, matched_ridge_model, pd):
    matched_ridge_table = pd.DataFrame(
        [group_term_structure(matched_ridge_model, "v", "theta|participant_id")]
    )
    matched_ridge_table
    return (matched_ridge_table,)


@app.cell
def _(matched_ridge_model, model_graph):
    matched_ridge_graph = model_graph(matched_ridge_model)
    matched_ridge_graph
    return (matched_ridge_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Repeated group-only owners have the same ridge

    A different ambiguity appears when the same unmatched expression has free
    means under multiple grouping factors. Shifting every participant effect up
    and every `conf` effect down leaves the predictor unchanged. Explicit priors
    remain authoritative, so HSSM builds the centered model and emits one
    aggregated warning.
    """)
    return


@app.cell
def _(assert_connected, build_model, group_term_structure, hssm, pd):
    def _fresh_free_location():
        return hssm.Prior(
            "Normal",
            mu=hssm.Prior("Normal", mu=0.0, sigma=0.5),
            sigma=hssm.Prior("HalfNormal", sigma=0.5),
        )

    _repeated_owner_include = [
        {
            "name": "v",
            "formula": ("v ~ 1 + (0 + theta | participant_id) + (0 + theta | conf)"),
            "prior": {
                "theta|participant_id": _fresh_free_location(),
                "theta|conf": _fresh_free_location(),
            },
        }
    ]
    repeated_owner_model, repeated_owner_messages = build_model(
        _repeated_owner_include,
        a=1.5,
        noncentered=False,
    )
    assert len(repeated_owner_messages) == 1
    assert "identified only by the priors" in repeated_owner_messages[0]
    assert "theta|participant_id" in repeated_owner_messages[0]
    assert "theta|conf" in repeated_owner_messages[0]
    assert_connected(repeated_owner_model)

    repeated_owner_table = pd.DataFrame(
        [
            group_term_structure(
                repeated_owner_model,
                "v",
                _term,
            )
            for _term in ("theta|participant_id", "theta|conf")
        ]
    )
    repeated_owner_table
    return repeated_owner_messages, repeated_owner_model, repeated_owner_table


@app.cell(hide_code=True)
def _(mo, repeated_owner_messages):
    mo.md(f"""
    ```text
    {repeated_owner_messages[0]}
    ```

    A cleaner formula adds common `theta` and uses mean-zero deviations for both
    grouping factors. Alternatively, choose exactly one centered group term as
    the location owner and fix the other group location intentionally.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Configuration errors fail fast too

    Component dictionaries are keyed by HSSM parameter names. A typo does not
    silently fall back to a default: HSSM/Bambi list the valid names during
    construction.
    """)
    return


@app.cell
def _(expect_model_error, pd):
    _unknown_key_include = [{"name": "v", "formula": "v ~ 1 + (1 | participant_id)"}]
    unknown_key_error = expect_model_error(
        _unknown_key_include,
        a=1.5,
        noncentered={"vv": True},
    )
    assert "Unknown component name(s)" in unknown_key_error
    assert "['a', 't', 'v', 'z']" in unknown_key_error
    unknown_key_table = pd.DataFrame(
        [
            {
                "input": "noncentered={'vv': True}",
                "result": "ValueError at construction",
                "message": unknown_key_error,
            }
        ]
    )
    unknown_key_table
    return unknown_key_error, unknown_key_table


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Practical rules

    | Formula/prior situation | Recommended action |
    | --- | --- |
    | Common and group expressions match | Let the common term own the location; use group `mu=0`; choose centered or non-centered for sampling geometry. |
    | One generated group-only expression | Let that group distribution own the location; HSSM's safe prior centers it automatically and explains the fallback. |
    | Explicit group-only prior with a free location | Set that prior effectively centered so its `mu` is retained. |
    | Explicit free or nonzero `mu` under non-centering | Change the formula/prior ownership or center the term; HSSM rejects the incompatible specification. |
    | Two free centered owners for one expression | Add the exact common term and use zero-mean deviations, or choose exactly one owner. |

    Two final boundaries are worth remembering:

    - `noncentered` only affects group-specific terms. A component with no group
      term is unchanged.
    - HSSM's truncated group-prior wrapper is incompatible with Bambi's
      hierarchical group-prior contract under either parameterization. Use an
      untruncated hierarchical coefficient prior and a support-respecting link.

    Continue with:

    - [Specify hierarchical group priors](https://lnccbrown.github.io/HSSM/how_to/specify_group_priors/) for the exact compatibility and location-ownership rules;
    - [Link functions and safe priors](https://lnccbrown.github.io/HSSM/tutorials/link_functions/) for predictor versus response scale; and
    - [Centered vs. non-centered parameterizations](https://lnccbrown.github.io/HSSM/tutorials/centered_vs_noncentered_basic_logic/) for the underlying graph transformation.

    For the sampling geometry behind this choice, Michael Betancourt's
    [Hierarchical Modeling](https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html)
    case study provides a detailed treatment.
    """)
    return


if __name__ == "__main__":
    app.run()
