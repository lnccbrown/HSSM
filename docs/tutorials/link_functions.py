# /// script
# requires-python = ">=3.12,<3.15"
# dependencies = [
#     "bambi==0.20.0",
#     "graphviz==0.21",
#     "hssm @ git+https://github.com/lnccbrown/HSSM.git@a7f6892d387f4b19f35e5db01f648abbb8535910",
#     "marimo==0.24.0",
#     "matplotlib==3.11.1",
#     "numpy==2.4.6",
#     "pandas==3.0.5",
#     "pymc==6.3.1",
# ]
# ///

"""Explain link functions, their HSSM role, and link-aware safe priors.

This construction-only marimo tutorial introduces link functions from first
principles, compares HSSM's identity and ``log_logit`` settings, and verifies
the current link-aware safe-prior and group-location behavior. No sampling is
required.

Run the pinned standalone environment locally or in Molab::

    uvx marimo edit --sandbox docs/tutorials/link_functions.py

To exercise an active HSSM checkout instead, ignore the inline environment::

    uv run --group notebook --group docs marimo edit --no-sandbox \
        docs/tutorials/link_functions.py
    uv run --group notebook --group docs marimo check --strict \
        docs/tutorials/link_functions.py
    uv run --group notebook --group docs marimo export html --no-sandbox \
        docs/tutorials/link_functions.py \
        --output /tmp/link-functions.html --force
    uv run --group notebook --group docs marimo export ipynb --no-sandbox \
        docs/tutorials/link_functions.py \
        --output docs/tutorials/link_functions.ipynb \
        --include-outputs --force
    uv run ruff format docs/tutorials/link_functions.ipynb
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
    from io import BytesIO, StringIO
    from tempfile import gettempdir

    # This notebook does not use GPU computation. Molab can expose an unusable
    # CUDA plugin, so keep JAX on CPU before importing the HSSM stack.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    os.environ.setdefault("MPLCONFIGDIR", f"{gettempdir()}/hssm-link-matplotlib")

    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)

    import bambi as bmb
    import marimo as mo
    import numpy as np
    import pandas as pd
    import pymc as pm
    import pytensor.tensor as pt

    with redirect_stderr(StringIO()):
        import matplotlib.pyplot as plt

    import hssm

    logging.getLogger("hssm").setLevel(logging.ERROR)
    hssm.set_floatX("float64")
    pd.set_option("display.max_colwidth", 80)
    return BytesIO, StringIO, bmb, hssm, mo, np, pd, plt, pm, pt, redirect_stdout


@app.cell(hide_code=True)
def _(bmb, hssm, mo):
    mo.md(f"""
    # Understanding link functions in HSSM

    A link function is the bridge between an ordinary regression and a model
    parameter that may be positive, bounded, or otherwise constrained. This
    tutorial starts from that idea and then shows how links determine the scale
    on which HSSM interprets regression coefficients and chooses safe priors.

    The examples use **HSSM {hssm.__version__}** and **Bambi {bmb.__version__}**.
    They only construct models and inspect their structure; no MCMC is needed.

    By the end, you should be able to answer four questions:

    1. What does a link do to a linear predictor?
    2. When should an HSSM parameter use identity, log, or generalized logit?
    3. Why do HSSM's safe intercept priors depend on the effective link?
    4. Who owns a population location in a hierarchical regression?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. A regression produces an unconstrained linear predictor

    For a predictor `x`, a simple regression first constructs

    \[
    \eta_i = \beta_0 + \beta_1 x_i.
    \]

    The number \(\eta_i\) can be anywhere on the real line. That is fine for an
    unbounded parameter such as drift rate \(v\), but it can be invalid for a
    boundary separation \(a>0\) or a starting-point fraction \(0<z<1\).

    A link \(g\) connects the parameter scale to the predictor scale. HSSM uses
    the inverse link when building the model:

    \[
    \theta_i = g^{-1}(\eta_i).
    \]

    The regression coefficients live on the **linear-predictor scale**. The
    inverse link turns their result into a valid value on the **parameter scale**.
    """)
    return


@app.cell
def _(mo, pd):
    parameter_support_table = pd.DataFrame(
        [
            {
                "DDM parameter": "v (drift rate)",
                "support": "(-inf, inf)",
                "support-respecting link": "identity",
                "inverse link": "v = eta",
            },
            {
                "DDM parameter": "a (boundary separation)",
                "support": "(0, inf)",
                "support-respecting link": "log",
                "inverse link": "a = exp(eta)",
            },
            {
                "DDM parameter": "z (starting-point fraction)",
                "support": "(0, 1)",
                "support-respecting link": "generalized logit",
                "inverse link": "z = 1 / (1 + exp(-eta))",
            },
        ]
    )
    parameter_support_table
    return (parameter_support_table,)


@app.cell
def _(BytesIO, mo, np, plt):
    eta_grid = np.linspace(-3.0, 3.0, 301)
    link_curves_figure, _axes = plt.subplots(1, 3, figsize=(11, 3.4))
    _curves = [
        ("Identity: unbounded", eta_grid, (-3.2, 3.2), "parameter = eta"),
        ("Log: positive", np.exp(eta_grid), (0.0, 20.5), "parameter = exp(eta)"),
        (
            "Generalized logit: (0, 1)",
            1.0 / (1.0 + np.exp(-eta_grid)),
            (-0.03, 1.03),
            "parameter = sigmoid(eta)",
        ),
    ]
    for _axis, (_title, _values, _limits, _label) in zip(_axes, _curves, strict=True):
        _axis.plot(eta_grid, _values, color="#3366cc", linewidth=2.5)
        _axis.axvline(0.0, color="0.65", linestyle="--", linewidth=1)
        _axis.scatter([0.0], [_values[len(_values) // 2]], color="#cc5533", zorder=3)
        _axis.set_title(_title)
        _axis.set_xlabel("linear predictor eta")
        _axis.set_ylabel(_label)
        _axis.set_ylim(*_limits)
        _axis.grid(alpha=0.2)
    link_curves_figure.tight_layout()
    _plot_buffer = BytesIO()
    link_curves_figure.savefig(_plot_buffer, format="png", dpi=140, bbox_inches="tight")
    link_curves_image = mo.image(
        _plot_buffer.getvalue(),
        alt="Identity, log, and generalized-logit inverse-link curves",
    )
    link_curves_image
    return eta_grid, link_curves_figure, link_curves_image


@app.cell
def _(mo):
    mo.md("""
    The three panels use the same horizontal predictor scale but different
    parameter scales:

    - **Identity** leaves the predictor unchanged and does not impose a bound.
    - **Log** maps every real predictor to a positive value.
    - **Generalized logit** maps every real predictor into finite lower and upper
      bounds. For `(0, 1)`, it is the familiar logistic sigmoid.

    Notice the red points at `eta = 0`: identity maps zero to `0`, log maps it to
    `1`, and generalized logit maps it to the midpoint `0.5`. This fact matters
    when interpreting intercept priors.
    """)
    return


@app.cell
def _(np, pd):
    example_x = np.array([-1.0, 0.0, 1.0])
    example_eta = 0.2 + 0.5 * example_x
    coefficient_example_table = pd.DataFrame(
        {
            "x": example_x,
            "eta = 0.2 + 0.5*x": example_eta,
            "identity parameter": example_eta,
            "positive parameter (log link)": np.exp(example_eta),
            "bounded parameter (0, 1)": 1.0 / (1.0 + np.exp(-example_eta)),
        }
    ).round(3)
    coefficient_example_table
    return coefficient_example_table, example_eta, example_x


@app.cell
def _(mo):
    mo.md("""
    The coefficients in the table are identical in every column. What changes is
    their parameter-scale meaning. The inverse link transforms the **complete**
    predictor `0.2 + 0.5*x`; it does not transform the intercept and slope
    separately. With a non-identity link, a one-unit change in `x` is additive on
    `eta`, not on the final DDM parameter.
    """)
    return


@app.cell
def _(np, pd):
    eta_landmarks = np.array([-2.0, 0.0, 2.0])
    landmark_table = pd.DataFrame(
        {
            "eta": eta_landmarks,
            "identity": eta_landmarks,
            "log": np.exp(eta_landmarks),
            "generalized logit (0, 1)": 1.0 / (1.0 + np.exp(-eta_landmarks)),
        }
    ).round(3)
    landmark_table
    return eta_landmarks, landmark_table


@app.cell
def _(mo):
    mo.md("""
    These three landmarks make the scale change concrete. In the
    [Molab notebook](https://molab.marimo.io/github/lnccbrown/HSSM/blob/main/docs/tutorials/link_functions.py),
    edit `eta_landmarks` and rerun the cell to explore other values.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Where the link sits in an HSSM regression

    In an HSSM formula such as `a ~ 1 + x`, Bambi constructs the intercept and
    slope, combines them into `eta`, and the model family applies the inverse
    link before passing trial-wise `a` values to the likelihood.

    HSSM exposes two model-level choices for regression parameters:

    - `link_settings=None` (the default) uses identity unless a parameter says
      otherwise;
    - `link_settings="log_logit"` keeps unbounded parameters on identity, uses
      log for positive parameters, and uses HSSM's generalized logit for
      parameters with finite lower and upper bounds.

    You can override the link for an individual regression by adding `"link"`
    to that parameter's `include` specification.
    """)
    return


@app.cell
def _(StringIO, hssm, np, pd, redirect_stdout):
    tutorial_data = pd.DataFrame(
        {
            "rt": 0.42 + 0.015 * np.arange(12),
            "response": np.where(np.arange(12) % 2, 1, -1),
            "x": np.linspace(-1.0, 1.0, 12),
            "participant_id": np.repeat(np.arange(4), 3),
            "item_id": np.tile(np.arange(3), 4),
        }
    )
    regression_specs = [
        {"name": _parameter, "formula": f"{_parameter} ~ 1 + x"}
        for _parameter in ("v", "a", "z", "t")
    ]
    model_kwargs = {
        "data": tutorial_data,
        "model": "ddm",
        "loglik_kind": "analytical",
        "include": regression_specs,
        "p_outlier": 0.0,
        "prior_settings": "safe",
        "process_initvals": False,
        "initval_jitter": 0.0,
    }

    def build_silent_model(**kwargs):
        """Construct an HSSM model without its initialization status message."""
        with redirect_stdout(StringIO()):
            return hssm.HSSM(**kwargs)

    tutorial_data.head()
    return build_silent_model, model_kwargs, regression_specs, tutorial_data


@app.cell
def _(build_silent_model, model_kwargs):
    identity_model = build_silent_model(**model_kwargs, link_settings=None)
    transformed_model = build_silent_model(**model_kwargs, link_settings="log_logit")
    return identity_model, transformed_model


@app.cell
def _(bmb, identity_model, np, pd, transformed_model):
    def link_name(link):
        """Return a consistent display name for string and object links."""
        return link if isinstance(link, str) else link.name

    def inverse_link_value(link, eta):
        """Map one predictor value back to its parameter scale."""
        _link = bmb.Link(link) if isinstance(link, str) else link
        return float(np.asarray(_link.linkinv(eta)))

    model_link_table = pd.DataFrame(
        [
            {
                "parameter": _parameter,
                "support": str(identity_model.params[_parameter].bounds),
                "default link": link_name(identity_model.params[_parameter].link),
                "log_logit preset": link_name(
                    transformed_model.params[_parameter].link
                ),
                "preset maps eta=0 to": inverse_link_value(
                    transformed_model.params[_parameter].link, 0.0
                ),
            }
            for _parameter in ("v", "a", "z", "t")
        ]
    )
    assert model_link_table.set_index("parameter")["log_logit preset"].to_dict() == {
        "v": "identity",
        "a": "log",
        "z": "gen_logit",
        "t": "log",
    }
    model_link_table
    return inverse_link_value, link_name, model_link_table


@app.cell
def _(mo):
    mo.md("""
    For this analytical DDM, the preset follows support rather than parameter names: an unbounded
    parameter remains on identity, a lower-bounded positive parameter uses log,
    and a parameter with two finite bounds uses generalized logit.

    The configured likelihood supplies those bounds. A likelihood with finite
    training bounds can therefore resolve a parameter differently—for example,
    a finitely bounded neural `v` can receive generalized logit. The preset uses
    log only for the exact `(0, inf)` shape; unusual one-sided bounds are left on
    identity with a warning. Fixed or otherwise non-regression parameters do not
    have a linear predictor and are not assigned links by this preset.

    With HSSM's identity default, coefficients keep a direct response-scale
    interpretation and safe intercepts can retain HDDM-derived priors. The
    `log_logit` preset is an explicit alternative that often gives hierarchical
    regressions an unconstrained coefficient space. Neither spelling is
    universally preferable; the link is part of the scientific model.
    """)
    return


@app.cell
def _(identity_model, pd, transformed_model):
    def format_prior(prior):
        """Format a Bambi prior compactly for a comparison table."""
        return str(prior)

    prior_scale_table = pd.DataFrame(
        [
            {
                "parameter": _parameter,
                "identity intercept prior": format_prior(
                    identity_model.params[_parameter].prior["Intercept"]
                ),
                "log_logit link": (
                    transformed_model.params[_parameter].link
                    if isinstance(transformed_model.params[_parameter].link, str)
                    else transformed_model.params[_parameter].link.name
                ),
                "log_logit intercept prior": format_prior(
                    transformed_model.params[_parameter].prior["Intercept"]
                ),
                "slope prior in both models": format_prior(
                    transformed_model.params[_parameter].prior["x"]
                ),
            }
            for _parameter in ("v", "a", "z", "t")
        ]
    )

    assert identity_model.params["a"].prior["Intercept"].name == "Gamma"
    assert identity_model.params["z"].prior["Intercept"].name == "Beta"
    assert transformed_model.params["a"].prior["Intercept"].name == "Normal"
    assert transformed_model.params["z"].prior["Intercept"].name == "Normal"
    assert all(
        format_prior(identity_model.params[_parameter].prior["x"])
        == format_prior(transformed_model.params[_parameter].prior["x"])
        for _parameter in ("v", "a", "z", "t")
    )
    prior_scale_table
    return format_prior, prior_scale_table


@app.cell
def _(mo):
    mo.md("""
    ## 3. The link determines the scale of a safe intercept prior

    This is the core logic behind HSSM's generated `prior_settings="safe"`
    defaults:

    - With an **identity link**, the intercept is itself a value of the DDM
      parameter. HSSM can therefore use its response-scale HDDM prior, such as a
      positive `Gamma` for `a` or a `(0, 1)` `Beta` for `z` in the analytical
      DDM used here.
    - With a **transformed link**, the intercept is an unconstrained coefficient
      inside `eta`. HSSM uses `Normal(0, 0.25)` on that coefficient scale. The
      inverse link then maps it into valid parameter values.
    - Ordinary slopes are coefficients on `eta` in either case, so their safe
      default remains `Normal(0, 0.25)`.
    - An explicitly supplied prior always takes precedence over a generated
      safe prior.

    The HDDM-derived response-scale intercepts shown here apply to analytical
    and black-box `ddm`, `ddm_sdv`, and `full_ddm` regressions. Neural
    `approx_differentiable` likelihoods use generic safe priors informed by their
    configured training bounds. Setting `prior_settings=None` does not remove
    priors; it delegates missing regression-term priors to Bambi, while simple
    HSSM parameters retain their configured defaults.

    For example, `a_Intercept = 0` under a log link means `a = exp(0) = 1`.
    It would be a mistake to truncate that coefficient at `a`'s response-scale
    lower bound: the transform already enforces positivity.

    Conversely, an identity-link prior that keeps the **intercept** inside a
    parameter's bounds cannot guarantee that every trial-wise value remains
    valid once slopes or group effects are added. A support-respecting link is
    the structural way to enforce support on the full predictor.
    """)
    return


@app.cell
def _(bmb, build_silent_model, format_prior, hssm, model_kwargs, pd):
    identity_link_specs = [
        ("link omitted", None),
        ('link="identity"', "identity"),
        ('bambi.Link("identity")', bmb.Link("identity")),
        ('hssm.Link("identity")', hssm.Link("identity")),
    ]
    _base_kwargs = {
        _key: _value for _key, _value in model_kwargs.items() if _key != "include"
    }
    _identity_rows = []
    for _label, _link in identity_link_specs:
        _a_spec = {"name": "a", "formula": "a ~ 1 + x"}
        if _link is not None:
            _a_spec["link"] = _link
        _model = build_silent_model(**_base_kwargs, include=[_a_spec])
        _identity_rows.append(
            {
                "user spelling": _label,
                "effective link": (
                    _model.params["a"].link
                    if isinstance(_model.params["a"].link, str)
                    else _model.params["a"].link.name
                ),
                "safe a intercept prior": format_prior(
                    _model.params["a"].prior["Intercept"]
                ),
            }
        )

    identity_equivalence_table = pd.DataFrame(_identity_rows)
    assert identity_equivalence_table["effective link"].eq("identity").all()
    assert identity_equivalence_table["safe a intercept prior"].nunique() == 1
    identity_equivalence_table
    return identity_equivalence_table, identity_link_specs


@app.cell
def _(mo):
    mo.md("""
    The four rows are semantically the same model and now receive the same safe
    response-scale prior. HSSM decides from the **effective link**, not from
    whether the user happened to omit a value or wrap it in a Link object.
    Omitted identity and each explicit identity spelling are therefore
    interchangeable under the safe settings.
    """)
    return


@app.cell
def _(bmb, build_silent_model, format_prior, model_kwargs, pd):
    explicit_intercept_prior = bmb.Prior("StudentT", nu=4.0, mu=1.2, sigma=0.4)
    _explicit_base_kwargs = {
        _key: _value for _key, _value in model_kwargs.items() if _key != "include"
    }
    explicit_prior_model = build_silent_model(
        **_explicit_base_kwargs,
        include=[
            {
                "name": "a",
                "formula": "a ~ 1 + x",
                "prior": {"Intercept": explicit_intercept_prior},
            }
        ],
    )
    assert (
        explicit_prior_model.params["a"].prior["Intercept"] is explicit_intercept_prior
    )
    explicit_prior_table = pd.DataFrame(
        [
            {
                "term": "Intercept",
                "source": "explicit user prior",
                "resolved prior": format_prior(
                    explicit_prior_model.params["a"].prior["Intercept"]
                ),
            },
            {
                "term": "x",
                "source": "missing term filled by safe settings",
                "resolved prior": format_prior(
                    explicit_prior_model.params["a"].prior["x"]
                ),
            },
        ]
    )
    explicit_prior_table
    return explicit_intercept_prior, explicit_prior_model, explicit_prior_table


@app.cell
def _(mo):
    mo.md("""
    `prior_settings="safe"` fills gaps; it does not rewrite the explicit
    `StudentT` intercept. Here it supplies only the missing slope prior.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Group effects: who owns the population location?

    Links determine the scale on which a hierarchical effect lives, but first
    the formula must say **which term owns its population location**.

    A matching common and group expression is a population effect plus a
    deviation. For example,

    \[
    \eta_{ij}=(\beta_x+u_j)x_{ij},
    \qquad u_j\sim\mathcal N(0,\sigma_x).
    \]

    Here `x` owns the population effect and `x|participant_id` must have mean
    zero. It can use Bambi's non-centered `sigma * offset` form because zero is
    the intended location.

    If the common `x` is absent, one group term can instead own the location:

    \[
    \eta_{ij}=b_jx_{ij},
    \qquad b_j\sim\mathcal N(\mu_x,\sigma_x).
    \]

    Replacing this `mu` by zero changes the scientific model—it removes the
    population-level slope. HSSM therefore keeps the generated location-bearing
    hierarchy and centers that term. This term-level fallback preserves the
    requested population location: current Bambi non-centering would otherwise
    omit a nonzero or estimated group `mu`. For an explicit user prior, HSSM
    raises before model construction rather than silently building a different
    model.

    Finally, if the same unmatched expression appears under two grouping
    factors, neither has a unique claim to the location. In

    \[
    \eta_i=x_i\left(b^{(s)}_{s[i]}+b^{(r)}_{r[i]}\right),
    \]

    adding a constant to every subject effect and subtracting it from every
    item effect leaves `eta` unchanged. Safe generation rejects this ambiguity
    rather than choosing an order-dependent owner. Fully explicit centered priors
    are preserved, but HSSM warns when two or more free group means create this
    likelihood ridge: proper priors may identify the decomposition, while the data
    still identify only the sum.
    """)
    return


@app.cell
def _(mo, pd):
    group_ownership_table = pd.DataFrame(
        [
            {
                "formula pattern": "x + (0 + x | participant_id)",
                "population owner": "common x",
                "generated group location": "fixed at 0 (deviation)",
                "safe behavior": "honor requested centered/non-centered form",
            },
            {
                "formula pattern": "(0 + x | participant_id)",
                "population owner": "the one group distribution",
                "generated group location": "estimated",
                "safe behavior": "preserve it and center this term",
            },
            {
                "formula pattern": "(0 + x | subject) + (0 + x | item)",
                "population owner": "ambiguous",
                "generated group location": "two competing locations",
                "safe behavior": "reject and request an explicit owner",
            },
        ]
    )
    mo.Html(group_ownership_table.to_html(index=False, border=0))
    return (group_ownership_table,)


@app.cell
def _(build_silent_model, hssm, model_kwargs):
    _group_base_kwargs = {
        _key: _value for _key, _value in model_kwargs.items() if _key != "include"
    }
    _group_cases = {
        "identity HDDM location": {
            "parameter": "a",
            "formula": "a ~ 0 + (1 | participant_id)",
            "link": "identity",
            "fixed": {"v": 0.0, "z": 0.5, "t": 0.2},
        },
        "log-scale location": {
            "parameter": "a",
            "formula": "a ~ 0 + (1 | participant_id)",
            "link": "log",
            "fixed": {"v": 0.0, "z": 0.5, "t": 0.2},
        },
        "generalized-logit location": {
            "parameter": "z",
            "formula": "z ~ 0 + (1 | participant_id)",
            "link": hssm.Link("gen_logit", bounds=(0.0, 1.0)),
            "fixed": {"v": 0.0, "a": 1.5, "t": 0.2},
        },
        "matched non-centered deviation": {
            "parameter": "v",
            "formula": "v ~ 1 + x + (0 + x | participant_id)",
            "link": "identity",
            "fixed": {"a": 1.5, "z": 0.5, "t": 0.2},
        },
    }
    group_location_models = {}
    for _label, _case in _group_cases.items():
        _spec = {
            "name": _case["parameter"],
            "formula": _case["formula"],
            "link": _case["link"],
        }
        group_location_models[_label] = build_silent_model(
            **_group_base_kwargs,
            include=[_spec],
            noncentered=True,
            **_case["fixed"],
        )

    try:
        build_silent_model(
            **_group_base_kwargs,
            include=[
                {
                    "name": "v",
                    "formula": ("v ~ 1 + (0 + x | participant_id) + (0 + x | item_id)"),
                    "link": "identity",
                }
            ],
            noncentered=True,
            a=1.5,
            z=0.5,
            t=0.2,
        )
    except ValueError as _error:
        ambiguous_owner_error = str(_error)
    else:
        raise AssertionError("Ambiguous safe group locations were not rejected")

    assert "Multiple unmatched group-specific terms" in ambiguous_owner_error
    assert "x|participant_id" in ambiguous_owner_error
    assert "x|item_id" in ambiguous_owner_error
    return ambiguous_owner_error, group_location_models


@app.cell
def _(bmb, format_prior, group_location_models, link_name, mo, np, pd):
    _group_rows = []
    _group_specs = [
        (
            "identity HDDM location",
            "a",
            "1|participant_id",
            "response scale",
        ),
        ("log-scale location", "a", "1|participant_id", "log-predictor scale"),
        (
            "generalized-logit location",
            "z",
            "1|participant_id",
            "generalized-log-odds scale",
        ),
        (
            "matched non-centered deviation",
            "v",
            "x|participant_id",
            "identity predictor scale",
        ),
    ]
    for _label, _parameter, _term, _scale in _group_specs:
        _model = group_location_models[_label]
        _prior = _model.params[_parameter].prior[_term]
        _mu = _prior.args.get("mu")
        _mu_is_free = isinstance(_mu, bmb.Prior)
        _prefix = f"{_parameter}_{_term}"
        _free_vars = {_variable.name for _variable in _model.pymc_model.free_RVs}
        _group_rows.append(
            {
                "case": _label,
                "effective link": link_name(_model.params[_parameter].link),
                "group term": _term,
                "outer family": _prior.name,
                "location": format_prior(_mu) if _mu_is_free else repr(_mu),
                "location scale": _scale,
                "effective form": (
                    "centered location owner"
                    if _prior.noncentered is False
                    else "non-centered zero-mean deviation"
                ),
                "direct group RV": _prefix in _free_vars,
                "offset RV": f"{_prefix}_offset" in _free_vars,
            }
        )

    group_location_prior_table = pd.DataFrame(_group_rows)
    _identity_prior = (
        group_location_models["identity HDDM location"]
        .params["a"]
        .prior["1|participant_id"]
    )
    _log_prior = (
        group_location_models["log-scale location"]
        .params["a"]
        .prior["1|participant_id"]
    )
    _gen_logit_prior = (
        group_location_models["generalized-logit location"]
        .params["z"]
        .prior["1|participant_id"]
    )
    _matched_prior = (
        group_location_models["matched non-centered deviation"]
        .params["v"]
        .prior["x|participant_id"]
    )

    assert _identity_prior.name == "Gamma"
    assert _log_prior.name == "Normal"
    assert _gen_logit_prior.name == "Normal"
    assert _identity_prior.noncentered is False
    assert _log_prior.noncentered is False
    assert _gen_logit_prior.noncentered is False
    assert not isinstance(_matched_prior.args["mu"], bmb.Prior)
    assert np.all(np.asarray(_matched_prior.args["mu"]) == 0.0)
    assert group_location_prior_table.iloc[:3]["direct group RV"].all()
    assert not group_location_prior_table.iloc[:3]["offset RV"].any()
    assert not group_location_prior_table.iloc[3]["direct group RV"]
    assert group_location_prior_table.iloc[3]["offset RV"]
    mo.Html(group_location_prior_table.to_html(index=False, border=0))
    return (group_location_prior_table,)


@app.cell
def _(mo):
    mo.md("""
    The first three rows all have one unmatched group intercept and therefore
    one population-location owner. HSSM sets only those generated priors to
    `noncentered=False`, retaining a direct group random variable and its
    location hyperprior.

    Their links change what that location means:

    - under **identity**, the `a` hierarchy is the HDDM-derived response-scale
      `Gamma` hierarchy;
    - under **log**, `exp(mu)` is a reference median on the positive parameter
      scale, not the mean after integrating over group variation; and
    - under **generalized logit**, the inverse-linked `mu` is a bounded reference
      value, again not generally an expectation.

    The fourth row is different: common `x` owns the population slope, so the
    group term is a zero-mean deviation and can stay non-centered. Under a log
    link such a zero deviation would be a neutral multiplicative factor
    `exp(0)=1`; under generalized logit it would leave the common predictor
    unchanged. Zero never means that the final HSSM parameter is forced to zero.

    These ownership rules are identical for every link because all locations
    combine on `eta` before the inverse link. The inverse link also cannot repair
    two competing owners: it receives the same unchanged sum.
    """)
    return


@app.cell
def _(ambiguous_owner_error, mo, pd):
    ambiguous_owner_table = pd.DataFrame(
        [
            {
                "attempted formula": (
                    "v ~ 1 + (0 + x | participant_id) + (0 + x | item_id)"
                ),
                "result": "rejected before Bambi model construction",
                "reason": "two unmatched x terms compete for one population location",
                "recommended repair": (
                    "add common x and use zero-mean group deviations, or explicitly "
                    "choose exactly one location owner"
                ),
            }
        ]
    )
    assert "Add the exact common formula term" in ambiguous_owner_error
    mo.Html(ambiguous_owner_table.to_html(index=False, border=0))
    return (ambiguous_owner_table,)


@app.cell
def _(group_location_models, pm):
    def make_group_location_graph(case):
        """Render one construction-only PyMC graph for the selected case."""
        return pm.model_to_graphviz(
            group_location_models[case].pymc_model,
            graph_attr={"bgcolor": "white", "rankdir": "LR"},
        )

    return (make_group_location_graph,)


@app.cell
def _(mo):
    mo.md("""
    ### Identity-linked HDDM group location

    The direct `a_1|participant_id` node receives its population location and
    scale from the Gamma hierarchy. No offset node replaces that location.
    HSSM's likelihood bounds still apply, but finite coefficient bounds are not
    automatically propagated to generic identity-linked group priors. A bound on
    one coefficient would not constrain the complete predictor after other effects
    are added. Prefer a support-respecting transformed link when appropriate; the
    [group-prior guide](https://lnccbrown.github.io/HSSM/how_to/specify_group_priors/)
    explains the remaining identity-link choices and limitations.
    """)
    return


@app.cell
def _(make_group_location_graph):
    identity_group_location_graph = make_group_location_graph("identity HDDM location")
    identity_group_location_graph
    return


@app.cell
def _(mo):
    mo.md("""
    ### Log-linked group location

    This has the same ownership structure, but the Normal hierarchy now lives
    on the unconstrained log-predictor scale. The inverse link applies `exp` only
    after the participant-specific predictor is assembled.
    """)
    return


@app.cell
def _(make_group_location_graph):
    log_group_location_graph = make_group_location_graph("log-scale location")
    log_group_location_graph
    return


@app.cell
def _(mo):
    mo.md("""
    ### Generalized-logit group location

    The group hierarchy is again Normal on an unconstrained predictor. HSSM's
    bounded inverse link maps the assembled predictor into `(0, 1)`.
    """)
    return


@app.cell
def _(make_group_location_graph):
    gen_logit_group_location_graph = make_group_location_graph(
        "generalized-logit location"
    )
    gen_logit_group_location_graph
    return


@app.cell
def _(mo):
    mo.md("""
    ### Matched non-centered deviation

    Common `v_x` owns the population slope. The participant deviation has a
    scale and offset but no free group mean, which is the faithful non-centered
    representation for this matched case.
    """)
    return


@app.cell
def _(make_group_location_graph):
    matched_group_deviation_graph = make_group_location_graph(
        "matched non-centered deviation"
    )
    matched_group_deviation_graph
    return


@app.cell
def _(mo):
    mo.md("""
    ## 5. Built-in links versus custom links

    For built-in names such as `"identity"`, `"log"`, and `"logit"`, Bambi
    already knows all required numerical and symbolic functions. A custom link
    needs three related callables:

    1. `link`: numerical parameter-to-predictor mapping;
    2. `linkinv`: numerical predictor-to-parameter mapping, used outside the
       PyMC graph for operations such as prediction; and
    3. `linkinv_backend`: the symbolic inverse used to build the PyTensor graph.

    This is why NumPy functions are appropriate for the first two roles, while
    the backend inverse should be written with PyTensor operations.
    """)
    return


@app.cell
def _(build_silent_model, hssm, model_kwargs, np, pd, pt):
    custom_log_link = hssm.Link(
        "custom_log",
        link=np.log,
        linkinv=np.exp,
        linkinv_backend=pt.exp,
    )
    _response_values = np.array([0.5, 1.0, 2.0])
    _predictor_values = custom_log_link.link(_response_values)
    assert np.allclose(custom_log_link.linkinv(_predictor_values), _response_values)
    _symbolic_eta = pt.scalar("tutorial_eta")
    _symbolic_parameter = custom_log_link.linkinv_backend(_symbolic_eta)
    assert _symbolic_parameter.owner is not None

    _custom_base_kwargs = {
        _key: _value for _key, _value in model_kwargs.items() if _key != "include"
    }
    custom_link_model = build_silent_model(
        **_custom_base_kwargs,
        include=[
            {
                "name": "a",
                "formula": "a ~ 1 + x",
                "link": custom_log_link,
            }
        ],
    )
    assert custom_link_model.params["a"].link is custom_log_link
    assert custom_link_model.model.family.link["a"] is custom_log_link

    custom_link_roles = pd.DataFrame(
        [
            {
                "argument": "link=np.log",
                "direction": "parameter -> predictor",
                "execution": "numerical, outside the PyMC graph",
            },
            {
                "argument": "linkinv=np.exp",
                "direction": "predictor -> parameter",
                "execution": "numerical prediction and utilities",
            },
            {
                "argument": "linkinv_backend=pt.exp",
                "direction": "predictor -> parameter",
                "execution": "symbolic, inside the PyMC graph",
            },
        ]
    )
    custom_link_roles
    return custom_link_model, custom_link_roles, custom_log_link


@app.cell
def _(mo):
    mo.md("""
    HSSM's `gen_logit` is the special built-in used for arbitrary finite bounds.
    For example, `hssm.Link("gen_logit", bounds=(0.2, 0.8))` maps every real
    predictor into `(0.2, 0.8)`.
    """)
    return


@app.cell
def _(pd):
    link_choice_guide = pd.DataFrame(
        [
            {
                "parameter support": "unbounded",
                "usual choice": "identity",
                "eta=0 means": "parameter=0",
                "main tradeoff": "direct interpretation; no support transform",
            },
            {
                "parameter support": "positive",
                "usual choice": "log",
                "eta=0 means": "parameter=1",
                "main tradeoff": "valid positivity; multiplicative parameter effects",
            },
            {
                "parameter support": "finite (lower, upper)",
                "usual choice": "gen_logit",
                "eta=0 means": "parameter at midpoint",
                "main tradeoff": "valid bounds; nonlinear parameter effects",
            },
        ]
    )
    link_choice_guide
    return (link_choice_guide,)


@app.cell
def _(mo):
    mo.md("""
    ## 6. A practical workflow

    1. **Start from support.** Ask whether the parameter is unbounded, positive,
       or bounded on both sides.
    2. **Choose an interpretation.** Identity coefficients are additive on the
       parameter scale; transformed coefficients are additive on `eta` and
       nonlinear on the parameter scale.
    3. **Inspect the resolved model.** `model.params["a"].link` and
       `model.params["a"].prior` show what HSSM actually built.
    4. **Remember the prior scale.** A coefficient prior under a transformed
       link is not a prior directly on the DDM parameter.
    5. **Use prior predictive checks.** They reveal the parameter-scale behavior
       implied jointly by the coefficients and inverse link.

    An individual override looks like:

    ```python
    model = hssm.HSSM(
        data,
        include=[{"name": "a", "formula": "a ~ 1 + x", "link": "log"}],
        prior_settings="safe",
    )
    ```

    Or use `link_settings="log_logit"` to apply the support-aware preset to all
    regression parameters at once.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    ## Takeaway

    A link is not merely a numerical guardrail. It determines what regression
    coefficients mean, where priors live, and how an unconstrained predictor is
    translated into a valid SSM parameter. HSSM therefore treats every semantic
    spelling of identity alike, uses response-scale safe intercept priors for
    identity, and uses coefficient-scale priors for transformed links.

    A hierarchical formula adds a separate ownership decision. A matching common
    effect owns the population location and its group terms are zero-mean
    deviations. One unmatched group term may own the location, so its generated
    hierarchy is retained and centered. Several unmatched terms for the same
    expression are ambiguous and require an explicit model decision.

    Continue with:

    - the [`hssm.Link` API](https://lnccbrown.github.io/HSSM/api/link/) for supported and custom links;
    - [Specify priors and fix parameters](https://lnccbrown.github.io/HSSM/how_to/specify_priors/) for
      explicit prior control;
    - [Specify hierarchical group priors](https://lnccbrown.github.io/HSSM/how_to/specify_group_priors/) for
      location ownership, parameterization compatibility, and link-scale bounds;
    - [Set initial values](https://lnccbrown.github.io/HSSM/tutorials/initial_values/) for links and initialization;
    - [Random-slope prior diagnostics](https://lnccbrown.github.io/HSSM/tutorials/random_slope_safe_priors/) for
      matching common and group-specific regression terms;
    - [Coming from HDDM](https://lnccbrown.github.io/HSSM/explanations/coming_from_hddm/) for
      the group-only, location-bearing hierarchy used by HDDM-style models; and
    - Betancourt's [hierarchical modeling case study](https://betanalpha.github.io/assets/case_studies/hierarchical_modeling.html)
      for a deeper treatment of centered and non-centered posterior geometry.
    """)
    return


if __name__ == "__main__":
    app.run()
