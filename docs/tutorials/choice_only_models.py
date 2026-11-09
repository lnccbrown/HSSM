"""Simulate, fit and check choice-only softmax models with HSSM.

This marimo notebook is the source of truth for the output-bearing Jupyter
artifact published by MkDocs. Run a quick local check with::

    uv run --group notebook --group docs marimo check --strict \
        docs/tutorials/choice_only_models.py
    uv run --group notebook --group docs marimo export html --no-sandbox \
        docs/tutorials/choice_only_models.py \
        --output /tmp/choice-only-models.html --force

Regenerate the committed full-run artifact with::

    FULL_RUN=1 uv run --group notebook --group docs marimo export ipynb \
        --no-sandbox docs/tutorials/choice_only_models.py \
        --output docs/tutorials/choice_only_models.ipynb \
        --include-outputs --sort top-down --force
    uv run ruff format docs/tutorials/choice_only_models.ipynb
"""

# ruff: noqa: B018, D401, E501, PLR1711  (generated marimo notebook: prose, display expressions, and bare returns)
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Choice-only models

    Not all behavioral data comes with response times. In questionnaire studies, forced-choice paradigms without time pressure, or when RT is simply not recorded, we only observe **which option** a participant chose. HSSM supports this setting through **choice-only models**.

    This tutorial uses the softmax inverse-temperature models that ship with HSSM: `softmax_inv_temperature_2`, `_3` and `_4`, for two, three and four options.

    We will:

    1. Review the softmax inverse-temperature model and its likelihood
    2. Simulate choice data with `hssm.simulate_data`
    3. Fit simulated 2-choice data, examine what the data do (and do not) identify, and check the model's predictions
    4. Add a trial-level covariate that modulates choice
    5. Extend to a hierarchical (multi-subject) setting
    6. Demonstrate the 3-choice variant
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Run this tutorial

    <a href="https://colab.research.google.com/github/lnccbrown/HSSM/blob/main/docs/tutorials/choice_only_models.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open this notebook in Google Colab"></a>

    On Colab, run `%pip install hssm` in a new cell once, then restart the runtime. For local setup, GPU extras, and troubleshooting see the [Installation guide](https://lnccbrown.github.io/HSSM/getting_started/installation/).
    """)
    return


@app.cell
def _():
    import logging
    import os
    import warnings

    # This tutorial does not need a GPU. Keep headless documentation builds
    # away from host CUDA plugins before importing HSSM/JAX.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

    import arviz as az
    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from scipy.special import softmax

    import hssm

    hssm.set_floatX("float32", update_jax=True)

    FULL_RUN = os.environ.get("FULL_RUN") == "1"
    N_SAMPLES = 500 if FULL_RUN else 100
    return N_SAMPLES, az, hssm, mo, np, pd, plt, softmax


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Softmax Inverse-Temperature Model

    Consider a task with $K$ choice options. The model has the following parameters:

    - $\beta > 0$ — the **inverse temperature**, controlling how deterministic choices are.
    - $\ell_1, \dots, \ell_{K-1}$ — **logits** encoding relative preference for each option. The reference option has $\ell_0 = 0$.

    The probability of choosing option $k$ is:

    $$P(c = k \mid \beta, \boldsymbol{\ell}) = \frac{\exp(\beta \, \ell_k)}{\sum_{j=0}^{K-1} \exp(\beta \, \ell_j)}$$

    The log-likelihood for a single trial with observed choice $c_i$ is:

    $$\log P(c_i \mid \beta, \boldsymbol{\ell}) = \beta \, \ell_{c_i} - \log \sum_{j=0}^{K-1} \exp(\beta \, \ell_j)$$

    **Interpretation:**
    - As $\beta \to 0$, choices become **uniform** regardless of logits.
    - As $\beta \to \infty$, leads to **deterministically** picking the option with the highest logit.
    - Logits encode relative preference: $\ell_k > 0$ means option $k$ is preferred over the reference.

    Because only the choice is modeled, the data requires just a single `response` column — no `rt` column is needed. HSSM detects this automatically from the model configuration.

    ### A Note on Parameter Identifiability

    Notice that choice probabilities depend on the **products** $\beta \, \ell_k$, not on $\beta$ and $\ell_k$ individually. For any scalar $c > 0$, the reparameterization $\beta' = c \, \beta$, $\ell_k' = \ell_k / c$ yields identical choice probabilities. This means $\beta$ and the logits are not separately identifiable from choice data alone — only the composite quantities $\beta \, \ell_k$ are.

    In practice this manifests as a **ridge in the joint posterior**: $\beta$ and the logits trade off against each other, producing strong negative correlations, slow mixing (low ESS), and posteriors that are wider than one might expect. The pair plots in this tutorial will make this clearly visible.

    **Dealing with the tradeoff.** Two strategies handle the scale tradeoff — one removes the redundancy from the likelihood itself, the other sets the scale through the posterior:

    1. **Fix $\beta$** (e.g., $\beta = 1$) and let the logits absorb the full scale. This eliminates the redundancy entirely and is the simplest approach when only relative preferences matter.
    2. **Use informative priors** on $\beta$ or the logits to deliberately set the scale. The likelihood stays flat along the ridge, but the posterior no longer is.

    Two further changes are often expected to help, and we examine both in this tutorial — but neither removes the scale invariance, which holds exactly for any number of choices and with trial-level covariates (the reparameterization above simply rescales every logit or regression coefficient together):

    3. **Add more choices.** With $K \geq 3$ options the flat direction becomes a one-dimensional curve in a higher-dimensional parameter space; sampling geometry changes and diagnostics may improve, but $\beta' = c \, \beta$, $\ell_k' = \ell_k / c$ still leaves the likelihood untouched.
    4. **Introduce covariates** that modulate logits across trials. The composite products (e.g., $\beta \times$ intercept and $\beta \times$ slope) are identified and sampling can improve, but $\beta$ and the coefficients individually still trade off exactly.

    We will encounter this tradeoff throughout the tutorial. The summary to keep in mind: **the data identify only the composite products $\beta \, \ell_k$.** Fixing $\beta$ removes the redundancy from the likelihood; an informative prior leaves the likelihood flat but pins the scale through the posterior — individual-parameter inference then inherits that prior choice.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulating Choice Data

    `hssm.simulate_data` simulates choice-only models just like sequential sampling models: it calls the [ssm-simulators](https://github.com/lnccbrown/ssm-simulators) model of the same name. Like every simulator output, the result has an `rt` column; for a choice-only model it holds the placeholder `-1.0`, since no response time is simulated. The choice-only likelihood reads only `response`, so the placeholder can stay in the data.

    ### 2-Choice Data (Fixed Parameters)

    We generate 500 trials from a binary softmax model with known parameters. Its two options are coded `-1` and `1`.
    """)
    return


@app.cell
def _(hssm, mo):
    TRUE_BETA_BASIC = 2.0
    TRUE_LOGIT1_BASIC = 0.5

    data_2c = hssm.simulate_data(
        model="softmax_inv_temperature_2",
        theta={"beta": TRUE_BETA_BASIC, "logit1": TRUE_LOGIT1_BASIC},
        size=500,
        random_state=2025,
    )

    mo.md(data_2c.head().to_markdown(index=False))
    return TRUE_BETA_BASIC, TRUE_LOGIT1_BASIC, data_2c


@app.cell
def _(TRUE_BETA_BASIC, TRUE_LOGIT1_BASIC, data_2c, np, plt, softmax):
    # Sanity check: observed vs. analytic choice proportions
    _analytic = softmax(TRUE_BETA_BASIC * np.array([0.0, TRUE_LOGIT1_BASIC]))
    _observed = data_2c["response"].value_counts(normalize=True).sort_index()

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _x_pos = np.arange(2)
    _ax.bar(_x_pos - 0.15, _analytic, width=0.3, label="Analytic")
    _ax.bar(_x_pos + 0.15, _observed.values, width=0.3, label="Observed")
    _ax.set_xticks(_x_pos)
    _ax.set_xticklabels(_observed.index.astype(int))
    _ax.set_ylabel("P(choice)")
    _ax.set_xlabel("Response")
    _ax.legend()
    _ax.set_title("Choice proportions: analytic vs. observed")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 1: Basic 2-Choice Softmax

    We fit the simplest choice-only model. Only the `response` column enters the likelihood.
    """)
    return


@app.cell
def _(data_2c, hssm):
    model_basic = hssm.HSSM(
        data=data_2c,
        model="softmax_inv_temperature_2",
        loglik_kind="analytical",
    )

    print(model_basic)
    return (model_basic,)


@app.cell
def _(N_SAMPLES, model_basic):
    idata_basic = model_basic.sample(
        sampler="numpyro",
        chains=2,
        tune=N_SAMPLES,
        draws=N_SAMPLES,
        random_seed=2025,
        progressbar=False,
    )
    return (idata_basic,)


@app.cell
def _(az, idata_basic, mo):
    mo.md(az.summary(idata_basic).to_markdown(floatfmt=".4g"))
    return


@app.cell
def _(az, idata_basic):
    az.plot_trace_dist(idata_basic)
    return


@app.cell
def _(az, idata_basic):
    az.plot_dist(idata_basic, var_names=["beta", "logit1"], kind="hist")
    return


@app.cell
def _(az, idata_basic):
    az.plot_pair(idata_basic, var_names=["beta", "logit1"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Taking Stock

    The pair plot makes the **$\beta$–logit identifiability tradeoff** from the introduction concrete: the posterior concentrates along a ridge of constant $\beta \cdot \ell_1$ that passes through the true values ($\beta = 2.0$, $\ell_1 = 0.5$) — not around the true values themselves. Only the product is identified by the data; where the posterior sits *along* the ridge is determined by the priors. The divergences, $\hat{R}$ warnings, and low ESS in the summary above are diagnostics consistent with this flat direction (they are not unique to it, but here the pair plot shows its source).

    Note that the API is identical to RT-based HSSM models — only the `model` string and the data differ. In the next sections we will see how **covariates** (Model 2) and **more choices** (Model 4) change the sampling geometry — and why only **fixing $\beta$** (Model 3) actually resolves the degeneracy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Prior and Posterior Predictive Checks

    Because the softmax models simulate through ssm-simulators, `sample_prior_predictive` and `sample_posterior_predictive` work for choice-only models as they do for RT-based ones. HSSM's predictive plotting helpers do not support choice-only models yet, so we compare choice frequencies directly: each predictive draw is a full simulated data set of 500 choices, and for each draw we compute the proportion of `response = 1`. The histograms show how those proportions spread under the prior and under the posterior; the red line is the observed proportion.
    """)
    return


@app.cell
def _(data_2c, idata_basic, model_basic, plt):
    _prior = model_basic.sample_prior_predictive(draws=1000, random_seed=2025)
    _posterior = model_basic.sample_posterior_predictive(idata_basic, inplace=False)
    _observed = (data_2c["response"] == 1).mean()

    _fig, _axes = plt.subplots(1, 2, figsize=(8, 3), sharey=True)
    for _ax, _draws, _title in [
        (_axes[0], _prior["prior_predictive"], "Prior predictive"),
        (_axes[1], _posterior["posterior_predictive"], "Posterior predictive"),
    ]:
        _p1 = (_draws["response"] == 1).mean("__obs__").values.ravel()
        _ax.hist(_p1, bins=40, range=(0, 1))
        _ax.axvline(_observed, color="red", label="Observed")
        _ax.set_title(_title)
        _ax.set_xlabel("P(response = 1) in a draw")
    _axes[0].set_ylabel("Draws")
    _axes[1].legend()
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The prior predictive spreads over almost every possible proportion, with extra mass near 0 and 1 where the priors allow large products $\beta \cdot \ell_1$; the observed proportion sits well inside it. The posterior predictive concentrates tightly around the observed proportion. This is the identifiability story from the prediction side: the ridge in the pair plot leaves $\beta$ and $\ell_1$ individually uncertain, but every point on it implies nearly the same choice probability, so the predictions are sharp.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 2: Adding a Covariate

    In many experiments, stimulus or task properties modulate choice. Here we simulate a covariate `x` (think: reward difference, stimulus strength) that linearly affects `logit1`:

    $$\ell_{1,i} = \alpha + \gamma \, x_i$$

    with true values $\alpha = 0.3$ (intercept) and $\gamma = 0.8$ (slope). Passing one `logit1` value per trial with `size=1` simulates one choice per trial.
    """)
    return


@app.cell
def _(hssm, mo, np):
    TRUE_BETA_REG = 2.0
    TRUE_INTERCEPT = 0.3
    TRUE_SLOPE = 0.8

    x = np.random.default_rng(42).standard_normal(1000)

    data_reg = hssm.simulate_data(
        model="softmax_inv_temperature_2",
        theta={"beta": TRUE_BETA_REG, "logit1": TRUE_INTERCEPT + TRUE_SLOPE * x},
        size=1,
        random_state=42,
    ).assign(x=x)

    mo.md(data_reg.head().to_markdown(index=False))
    return (data_reg,)


@app.cell
def _(data_reg, hssm):
    model_reg = hssm.HSSM(
        data=data_reg,
        model="softmax_inv_temperature_2",
        loglik_kind="analytical",
        include=[{"name": "logit1", "formula": "logit1 ~ x"}],
    )

    print(model_reg)
    return (model_reg,)


@app.cell
def _(N_SAMPLES, model_reg):
    idata_reg = model_reg.sample(
        sampler="numpyro",
        chains=2,
        tune=N_SAMPLES,
        draws=N_SAMPLES,
        random_seed=2025,
        progressbar=False,
    )
    return (idata_reg,)


@app.cell
def _(az, idata_reg, mo):
    mo.md(az.summary(idata_reg).to_markdown(floatfmt=".4g"))
    return


@app.cell
def _(az, idata_reg):
    az.plot_trace_dist(idata_reg)
    return


@app.cell
def _(az, idata_reg):
    az.plot_dist(
        idata_reg, var_names=["logit1_Intercept", "logit1_x", "beta"], kind="hist"
    )
    return


@app.cell
def _(az, idata_reg):
    az.plot_pair(idata_reg, var_names=["beta", "logit1_x", "logit1_Intercept"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Predicted Choice Probability vs. Covariate

    We can visualize how the recovered regression translates into choice probability as a function of `x`.
    """)
    return


@app.cell
def _(data_reg, idata_reg, np, pd, plt, softmax):
    # Posterior mean parameters
    _post = idata_reg["posterior"]
    _beta_hat = float(_post["beta"].mean())
    _intercept_hat = float(_post["logit1_Intercept"].mean())
    _slope_hat = float(_post["logit1_x"].mean())

    # Predicted P(response=1) as a function of x
    _x_grid = np.linspace(-3, 3, 200)
    _logit1_grid = _intercept_hat + _slope_hat * _x_grid
    _p1_grid = softmax(
        _beta_hat * np.column_stack([np.zeros_like(_x_grid), _logit1_grid]),
        axis=1,
    )[:, 1]

    # Binned observed proportions
    _obs_props = data_reg.groupby(pd.cut(data_reg["x"], bins=15), observed=True)[
        "response"
    ].apply(lambda s: (s == 1).mean())
    _bin_centers = _obs_props.index.map(lambda iv: iv.mid)

    _fig, _ax = plt.subplots(figsize=(6, 4))
    _ax.scatter(_bin_centers, _obs_props.values, label="Observed (binned)", zorder=3)
    _ax.plot(_x_grid, _p1_grid, color="tab:orange", label="Posterior mean prediction")
    _ax.set_xlabel("x (covariate)")
    _ax.set_ylabel("P(response = 1)")
    _ax.legend()
    _ax.set_title("Choice probability vs. covariate")
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Taking Stock

    The predicted choice-probability curve tracks the observed data closely — the curve depends only on the identified products, so it is well determined even though the individual parameters are not. Regression on choice-only model parameters works exactly as for RT-based HSSM models — via the `include` argument.

    The identifiability tradeoff itself is fully intact: $\beta$ is estimated well above its true value (around 5 vs. 2.0) while the intercept and slope on `logit1` are proportionally scaled down — the posterior simply settled at a different point on the same ridge. Trial-level covariates do **not** constrain the scale: rescaling $\beta$ against both regression coefficients together leaves the likelihood unchanged, so what the data identify are the products $\beta \times$ intercept and $\beta \times$ slope. Strategy 4 changes the sampling geometry — ESS is higher than for Model 1 — but not what is identified. In the hierarchical model next, we fix $\beta$ to eliminate the redundancy entirely.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 3: Hierarchical (Multi-Subject)

    With multiple subjects, we can introduce a hierarchy: subject-level logits are drawn from a shared group distribution, providing regularization.

    We simulate 20 subjects (200 trials each), with subject-level $\ell_1^{(s)} \sim \mathcal{N}(0.5, 0.3)$ and a shared $\beta = 2.0$. As in Model 2, we pass one `logit1` value per trial: each subject's value, repeated for that subject's 200 trials.
    """)
    return


@app.cell
def _(hssm, mo, np):
    TRUE_BETA_HIER = 2.0
    TRUE_LOGIT1_MU = 0.5
    TRUE_LOGIT1_SD = 0.3
    N_SUBJECTS = 20
    N_TRIALS_HIER = 200

    subject_logit1 = np.random.default_rng(123).normal(
        TRUE_LOGIT1_MU, TRUE_LOGIT1_SD, size=N_SUBJECTS
    )

    data_hier = hssm.simulate_data(
        model="softmax_inv_temperature_2",
        theta={
            "beta": TRUE_BETA_HIER,
            "logit1": np.repeat(subject_logit1, N_TRIALS_HIER),
        },
        size=1,
        random_state=123,
    ).assign(participant_id=np.repeat(np.arange(N_SUBJECTS), N_TRIALS_HIER))

    print(f"Subject-level logit1 (true): {np.round(subject_logit1, 3)}")
    mo.md(data_hier.head().to_markdown(index=False))
    return TRUE_BETA_HIER, data_hier, subject_logit1


@app.cell
def _(data_hier, plt):
    # Quick look: choice proportions per subject
    _props = data_hier.groupby("participant_id")["response"].apply(
        lambda s: (s == 1).mean()
    )

    _fig, _ax = plt.subplots(figsize=(6, 3))
    _ax.bar(_props.index, _props.values)
    _ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    _ax.set_xlabel("Participant ID")
    _ax.set_ylabel("P(response = 1)")
    _ax.set_title("Observed choice proportion per subject")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(TRUE_BETA_HIER, data_hier, hssm):
    model_hier = hssm.HSSM(
        data=data_hier,
        model="softmax_inv_temperature_2",
        loglik_kind="analytical",
        include=[
            {
                "name": "logit1",
                "formula": "logit1 ~ 1 + (1|participant_id)",
                "prior": {
                    "1|participant_id": {
                        "name": "Normal",
                        "mu": 0,
                        "sigma": {
                            "name": "Weibull",
                            "alpha": 1.0,
                            "beta": 1.0,
                        },
                    },
                    "Intercept": {
                        "name": "Normal",
                        "mu": 0,
                        "sigma": 5.0,
                    },
                },
            },
        ],
        beta=TRUE_BETA_HIER,  # fix parameter; release to check for trade-offs
        p_outlier=0.0,
        noncentered=True,
    )

    print(model_hier)
    return (model_hier,)


@app.cell
def _(N_SAMPLES, model_hier):
    idata_hier = model_hier.sample(
        sampler="numpyro",
        chains=2,
        tune=2 * N_SAMPLES,
        draws=2 * N_SAMPLES,
        random_seed=2025,
        progressbar=False,
    )
    return (idata_hier,)


@app.cell
def _(az, idata_hier, mo):
    # The non-centered parameterization also stores the standardized offsets
    # (`..._offset`); show the subject effects themselves instead.
    hier_vars = [
        "logit1_Intercept",
        "logit1_1|participant_id_sigma",
        "logit1_1|participant_id",
    ]
    mo.md(az.summary(idata_hier, var_names=hier_vars).to_markdown(floatfmt=".4g"))
    return (hier_vars,)


@app.cell
def _(az, hier_vars, idata_hier):
    az.plot_trace_dist(idata_hier, var_names=hier_vars)
    return


@app.cell
def _(az, idata_hier):
    az.plot_forest(idata_hier, var_names=["logit1_1|participant_id"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Parameter Recovery: Subject-Level Logits

    We compare the recovered subject-level `logit1` values (group intercept plus subject offset) against the true values used for simulation.
    """)
    return


@app.cell
def _(idata_hier, plt, subject_logit1):
    # The subject-level logit1 = group intercept + subject offset
    _post = idata_hier["posterior"]
    _recovered = (
        (_post["logit1_Intercept"] + _post["logit1_1|participant_id"])
        .mean(["chain", "draw"])
        .values
    )

    _fig, _ax = plt.subplots(figsize=(5, 5))
    _ax.scatter(subject_logit1, _recovered)
    _lims = [
        min(subject_logit1.min(), _recovered.min()) - 0.1,
        max(subject_logit1.max(), _recovered.max()) + 0.1,
    ]
    _ax.plot(_lims, _lims, "k--", linewidth=0.8, label="Identity")
    _ax.set_xlabel("True logit1")
    _ax.set_ylabel("Recovered logit1 (posterior mean)")
    _ax.set_title("Subject-level parameter recovery")
    _ax.legend()
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Taking Stock

    By **fixing $\beta$** to its true value (strategy 1 from the introduction), we completely eliminate the $\beta$–logit tradeoff. The results speak for themselves: no divergences, high ESS across all parameters, $\hat{R}$ within about 0.01 of 1, and clean recovery of both the group mean and subject-level logits. Compare this to Models 1 and 2, where $\beta$ was free and the sampler struggled with the resulting ridge in the posterior. When only relative preferences matter and absolute scale is not of interest, fixing $\beta$ is the simplest and most effective strategy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Model 4: Extension to 3 Choices

    HSSM also ships `softmax_inv_temperature_3` for tasks with three options (and `softmax_inv_temperature_4` for four). The API is identical — just change the model name. With 3 choices we have two logit parameters ($\ell_1$, $\ell_2$), both relative to the reference $\ell_0 = 0$. The options are coded `0`, `1` and `2`.
    """)
    return


@app.cell
def _(hssm, mo):
    TRUE_BETA_3C = 1.5
    TRUE_LOGIT1_3C = 0.8
    TRUE_LOGIT2_3C = -0.3

    data_3c = hssm.simulate_data(
        model="softmax_inv_temperature_3",
        theta={
            "beta": TRUE_BETA_3C,
            "logit1": TRUE_LOGIT1_3C,
            "logit2": TRUE_LOGIT2_3C,
        },
        size=800,
        random_state=2025,
    )

    mo.md(data_3c["response"].value_counts(normalize=True).sort_index().to_markdown())
    return TRUE_BETA_3C, TRUE_LOGIT1_3C, TRUE_LOGIT2_3C, data_3c


@app.cell
def _(TRUE_BETA_3C, TRUE_LOGIT1_3C, TRUE_LOGIT2_3C, data_3c, np, plt, softmax):
    # Analytic check
    _analytic = softmax(TRUE_BETA_3C * np.array([0.0, TRUE_LOGIT1_3C, TRUE_LOGIT2_3C]))
    _observed = data_3c["response"].value_counts(normalize=True).sort_index()

    _fig, _ax = plt.subplots(figsize=(4, 3))
    _x_pos = np.arange(3)
    _ax.bar(_x_pos - 0.15, _analytic, width=0.3, label="Analytic")
    _ax.bar(_x_pos + 0.15, _observed.values, width=0.3, label="Observed")
    _ax.set_xticks(_x_pos)
    _ax.set_xticklabels([0, 1, 2])
    _ax.set_ylabel("P(choice)")
    _ax.set_xlabel("Response")
    _ax.legend()
    _ax.set_title("3-choice: analytic vs. observed")
    plt.tight_layout()
    _fig
    return


@app.cell
def _(data_3c, hssm):
    model_3c = hssm.HSSM(
        data=data_3c,
        model="softmax_inv_temperature_3",
        loglik_kind="analytical",
    )

    print(model_3c)
    return (model_3c,)


@app.cell
def _(N_SAMPLES, model_3c):
    idata_3c = model_3c.sample(
        sampler="numpyro",
        chains=2,
        tune=N_SAMPLES,
        draws=N_SAMPLES,
        random_seed=2025,
        progressbar=False,
    )
    return (idata_3c,)


@app.cell
def _(az, idata_3c, mo):
    mo.md(az.summary(idata_3c).to_markdown(floatfmt=".4g"))
    return


@app.cell
def _(az, idata_3c):
    az.plot_trace_dist(idata_3c)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Taking Stock

    What the 3-choice model recovers are the composite products $\beta \cdot \ell_1$ and $\beta \cdot \ell_2$ — adding options does **not** break the exact scale degeneracy: rescaling $\beta$ against both logits together leaves the likelihood unchanged, for any $K$. The sampler still warns about $\hat{R}$ and low ESS: the flat direction is still there. Any improvement in the diagnostics over the 2-choice Model 1 would reflect prior information and a changed posterior geometry (the flat direction is now a single curve in a larger parameter space), not restored identifiability.

    The API is unchanged — switching from 2 to 3 choices only requires changing the model string to `"softmax_inv_temperature_3"` and providing data with the appropriate choice labels (`[0, 1, 2]`). Recall that $\ell_0 = 0$ serves as the reference: all estimated logits are relative to option 0. For inference on individual parameters in practice, pin the scale — fix $\beta$ or use a deliberately informative prior — and combine that with the richer signal from multiple options as needed.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Related Tutorials

    - [Main Tutorial](https://lnccbrown.github.io/HSSM/tutorials/main_tutorial/) — one guided regression workflow with RT-based models
    - [Scientific Workflow](https://lnccbrown.github.io/HSSM/tutorials/scientific_workflow_hssm/) — iterative model building with real data
    - [RLSSM Basic Tutorial](https://lnccbrown.github.io/HSSM/tutorials/rlssm_basic/) — reinforcement learning sequential sampling models in HSSM
    - [Choice-only RLSSM](https://lnccbrown.github.io/HSSM/tutorials/choice_only_rlssm/) — when the choices come from *learning*: Rescorla–Wagner + softmax, no RT
    """)
    return


if __name__ == "__main__":
    app.run()
