"""Use HSSM's analytical DDM likelihood as the decision process of an RLSSM.

This marimo notebook is the source of truth for the output-bearing Jupyter
artifact published by MkDocs. Run a quick local check with::

    uv run --group notebook --group docs marimo check --strict \
        docs/how_to/rlssm_analytical_ddm.py
    uv run --group notebook --group docs marimo export html --no-sandbox \
        docs/how_to/rlssm_analytical_ddm.py \
        --output /tmp/rlssm-analytical-ddm.html --force

Regenerate the committed full-run artifact with::

    FULL_RUN=1 uv run --group notebook --group docs marimo export ipynb \
        --no-sandbox docs/how_to/rlssm_analytical_ddm.py \
        --output docs/how_to/rlssm_analytical_ddm.ipynb \
        --include-outputs --force
    uv run ruff format docs/how_to/rlssm_analytical_ddm.ipynb
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

    # This how-to does not need a GPU. Keep Molab and headless documentation
    # builds away from host CUDA plugins before importing HSSM/JAX.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    # Silence only PyMC's notice that numpyro ignores the `init` HSSM passes;
    # every other warning, including sampling diagnostics, stays visible.
    warnings.filterwarnings(
        "ignore", message=r"`init='adapt_diag'` is ignored", category=UserWarning
    )
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

    import arviz as az
    import jax.numpy as jnp
    import marimo as mo
    import numpy as np
    import pytensor.tensor as pt
    from pymc.sampling.jax import get_jaxified_graph
    from pytensor.graph.rewriting.utils import rewrite_graph
    from ssms import rl

    import hssm
    from hssm.likelihoods.analytical import logp_ddm
    from hssm.utils import annotate_function

    # The analytical likelihood is validated in double precision, HSSM's default.
    hssm.set_floatX("float64")

    FULL_RUN = os.environ.get("FULL_RUN") == "1"
    return (
        FULL_RUN,
        annotate_function,
        az,
        get_jaxified_graph,
        hssm,
        jnp,
        logp_ddm,
        mo,
        np,
        pt,
        rewrite_graph,
        rl,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use the analytical DDM as an RLSSM decision process

    The built-in RLSSM presets evaluate their decision process with a likelihood
    approximation network (LAN). `"2AB_RW_DDM"`, for example, pairs a
    Rescorla–Wagner learner with the LAN for the DDM. This guide keeps that
    learner and replaces the LAN with HSSM's analytical DDM likelihood
    (Navarro & Fuss, 2009), the same `logp_ddm` that
    `hssm.HSSM(model="ddm", loglik_kind="analytical")` uses.

    It takes two steps:

    1. convert `logp_ddm` from PyTensor to JAX, because the RL likelihood
       builder composes the learner and the decision process in JAX; and
    2. swap that function into the preset's configuration.

    Everything else, including `hssm.rl.RLSSM`, sampling, and ArviZ, works as
    usual.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convert the analytical likelihood to JAX

    `get_jaxified_graph` turns a PyTensor graph into a JAX function. The
    rewrite turns `logp_ddm`'s parameter checks (`a >= 0`, `0 <= z <= 1`) into
    `-inf`, as PyMC does when it compiles a model; without it, the JAX version
    would skip them.

    The RL likelihood builder calls the decision process with one matrix that
    has a row per trial. Its columns follow the order of the function's
    `inputs`, declared in the next section.
    """)
    return


@app.cell
def _(get_jaxified_graph, jnp, logp_ddm, pt, rewrite_graph):
    _obs = pt.matrix("obs")  # columns: rt, response
    _v, _a, _z, _t = (pt.vector(name) for name in "vazt")
    _logp = rewrite_graph(
        logp_ddm(_obs, _v, _a, _z, _t),
        include=("canonicalize", "local_check_parameter_to_ninf_switch"),
    )
    _jax_logp = get_jaxified_graph(inputs=[_obs, _v, _a, _z, _t], outputs=[_logp])

    def ddm_logp(lan_matrix):
        """Per-trial analytical DDM log-likelihood; columns v, a, z, t, rt, response."""
        v, a, z, t, rt, response = (lan_matrix[:, i] for i in range(6))
        return _jax_logp(jnp.stack([rt, response], axis=1), v, a, z, t)[0]

    return (ddm_logp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A quick check that the JAX function returns what `logp_ddm` returns, on
    random parameters and response times from 1 ms to 10 s after `t`:
    """)
    return


@app.cell
def _(ddm_logp, logp_ddm, mo, np):
    _rng = np.random.default_rng(0)
    _n = 500
    _v = _rng.uniform(-3.0, 3.0, _n)
    _a = _rng.uniform(0.3, 2.5, _n)
    _z = _rng.uniform(0.1, 0.9, _n)
    _t = _rng.uniform(0.0, 1.0, _n)
    _rt = _t + 10 ** _rng.uniform(-3.0, 1.0, _n)
    _response = _rng.choice([-1.0, 1.0], _n)

    _jax_values = np.asarray(
        ddm_logp(np.column_stack([_v, _a, _z, _t, _rt, _response]))
    )
    _pytensor_values = logp_ddm(
        np.column_stack([_rt, _response]), _v, _a, _z, _t
    ).eval()
    mo.md(
        f"Largest absolute difference over {_n} trials: "
        f"`{np.max(np.abs(_jax_values - _pytensor_values)):.1e}`"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Swap the decision process of the preset

    Start from the `"2AB_RW_DDM"` configuration and replace only its
    `ssm_logp_func`. Passing the preset's `computed` mapping along keeps its
    Rescorla–Wagner learner, which computes the drift `v` on every trial. The
    bounds, the `feedback` column, and the `-1`/`1` response coding stay as the
    preset defines them.

    Leave `loglik_kind` as it is. `ssm_logp_func` already decides which
    likelihood the model uses, so changing `loglik_kind` would only relabel the
    model and change its default sampler.
    """)
    return


@app.cell
def _(annotate_function, ddm_logp, hssm, mo):
    rl_config = hssm.rl.get_rlssm_model_config("2AB_RW_DDM")
    rl_config.ssm_logp_func = annotate_function(
        inputs=["v", "a", "z", "t", "rt", "response"],
        outputs=["logp"],
        computed=rl_config.ssm_logp_func.computed,  # the preset's learner
    )(ddm_logp)
    mo.md(f"Sampled parameters: `{rl_config.list_params}`")
    return (rl_config,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulate a two-armed bandit

    `ssms.rl` simulates the same preset, so the data use the response coding
    and column names the configuration expects: `participant_id`, `rt`,
    `response` (`-1` or `1`), and `feedback`. Rows are grouped by participant,
    in trial order, and every participant has the same number of trials, as
    RLSSM requires. Keep the `-1`/`1` coding: the preset's learner maps it to
    the two arms itself.
    """)
    return


@app.cell
def _(mo, rl):
    true_params = {"rl_alpha": 0.3, "scaler": 3.0, "a": 1.2, "z": 0.5, "t": 0.3}
    simulator = rl.Simulator(rl.preset.get("2AB_RW_DDM"))
    data = simulator.simulate(
        theta=true_params, n_trials=200, n_participants=5, random_state=1
    )
    mo.md(data.head().to_markdown(index=False))
    return data, simulator, true_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit the model

    The simulated data contain no lapses, so we turn off HSSM's default 5%
    lapse component with `p_outlier=None`. With real data, keep it unless you
    have a reason not to.
    """)
    return


@app.cell
def _(FULL_RUN, data, hssm, rl_config):
    _draws = 1_000 if FULL_RUN else 300

    model = hssm.rl.RLSSM(data, model_config=rl_config, p_outlier=None)
    idata = model.sample(
        sampler="numpyro",
        chains=4,
        draws=_draws,
        tune=_draws,
        random_seed=1,
        progressbar=False,
    )
    return (idata,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare posteriors against the ground truth
    """)
    return


@app.cell
def _(az, idata, mo, true_params):
    _summary = az.summary(
        idata, var_names=list(true_params), ci_kind="hdi", ci_prob=0.94
    )
    _summary["true_value"] = [true_params[name] for name in _summary.index]
    _divergences = int(idata["sample_stats"]["diverging"].sum())
    mo.vstack([mo.md(_summary.to_markdown()), mo.md(f"Divergences: {_divergences}")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the fit by simulation

    `model.sample_posterior_predictive()` does not support RLSSM models yet,
    whichever likelihood they use. Simulate from `ssms.rl` instead: with
    `mode="ppc"`, the simulator replays the observed task, so the learner sees
    the same feedback as each participant did. Here we simulate once at the
    posterior means.
    """)
    return


@app.cell
def _(data, idata, mo, simulator, true_params):
    _posterior_means = {
        name: float(idata["posterior"][name].mean()) for name in true_params
    }
    _predicted = simulator.simulate(
        theta=_posterior_means, mode="ppc", observed_data=data, random_state=2
    )
    mo.md(
        "| | P(response = 1) | mean RT (s) |\n|---|---|---|\n"
        + "\n".join(
            f"| {_label} | {(_df['response'] == 1).mean():.3f} | {_df['rt'].mean():.3f} |"
            for _label, _df in [("observed", data), ("simulated", _predicted)]
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notes

    - **Why the analytical likelihood.** It is exact, and in our runs it sampled
      about three times faster than the LAN. It also has no training range: the
      learner's drift `v = scaler * (Q1 - Q0)` can reach ±10 within the preset's
      `scaler` bounds, while the DDM network was trained on `v` in (-3, 3).
    - **Simulated data.** `ssms` simulates with 1 ms Euler steps, which widens the
      effective boundary slightly, so the exact likelihood recovers `a` about
      0.02 above its generating value; the LAN, trained on the same simulator,
      does not show this. For tighter recovery checks, simulate with
      `delta_t=1e-4` in the preset's `ssm_kwargs`.
    - **Response coding.** Keep the preset's learner, as above. HSSM's own
      helper `hssm.rl.likelihoods.two_armed_bandit.compute_v_subject_wise`
      uses the response as an array index, so it gives wrong drifts for `-1`/`1`
      responses.
    - **Precision.** Keep HSSM's default double precision, as set above. The
      RLSSM tutorials switch to `float32`; with this converted likelihood that
      gives non-finite gradients for `a` and `z` on trials whose response time
      falls below `t`.
    - **The LAN still loads once.** `get_rlssm_model_config("2AB_RW_DDM")`
      loads the preset's network (and downloads it on first use) before you
      replace it.
    - **Native support is planned.** A built-in way to choose the analytical
      likelihood for an RLSSM decision process is planned in
      [lnccbrown/HSSM#1420](https://github.com/lnccbrown/HSSM/issues/1420).
    """)
    return


if __name__ == "__main__":
    app.run()
