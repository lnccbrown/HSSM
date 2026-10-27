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
    import time
    import warnings

    # This how-to does not need a GPU. Keep Molab and headless documentation
    # builds away from host CUDA plugins before importing HSSM/JAX.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    # Silence only PyMC's notice that numpyro ignores the `init` HSSM passes,
    # numpyro's notice that one CPU device runs the chains in sequence, and Bambi
    # deprecation notices raised inside HSSM itself; every other warning,
    # including sampling diagnostics, stays visible.
    warnings.filterwarnings(
        "ignore", message=r"`init='adapt_diag'` is ignored", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message=r"There are not enough devices", category=UserWarning
    )
    warnings.filterwarnings(
        "ignore", message=r".*is deprecated; use", category=FutureWarning
    )
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

    import arviz as az
    import marimo as mo
    import numpy as np
    import pytensor
    import pytensor.tensor as pt
    from ssms import rl

    import hssm
    from hssm.likelihoods import logp_ddm, logp_ddm_jax

    hssm.set_floatX("float64")

    FULL_RUN = os.environ.get("FULL_RUN") == "1"
    return (
        FULL_RUN,
        az,
        hssm,
        logp_ddm,
        logp_ddm_jax,
        mo,
        np,
        pt,
        pytensor,
        rl,
        time,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Use the analytical DDM as an RLSSM decision process

    The built-in RLSSM presets evaluate their decision process with a likelihood
    approximation network (LAN) by default. For the DDM you can ask for HSSM's
    analytical likelihood (Navarro & Fuss, 2009) instead:

    ```python
    hssm.rl.RLSSM(data, model="2AB_RW_DDM", decision_process_loglik_kind="analytical")
    ```

    This keeps the preset's Rescorla–Wagner learner and replaces only the
    decision likelihood. It is the same likelihood that
    `hssm.HSSM(model="ddm", loglik_kind="analytical")` uses, in the JAX form
    `hssm.likelihoods.logp_ddm_jax` that the RL likelihood needs. The LAN is not
    loaded at all. This guide checks that likelihood, fits a simulated two-armed
    bandit with it, and compares the fit with the LAN.

    Requires HSSM 0.5.1 or later.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The JAX likelihood

    `logp_ddm_jax` is a port of `logp_ddm` with the same parameterization,
    response coding and lower bound. On random parameters and response times
    from 1 ms to 10 s after `t`, the two agree to rounding error:
    """)
    return


@app.cell
def _(logp_ddm, logp_ddm_jax, mo, np, pt, pytensor):
    _rng = np.random.default_rng(0)
    _n = 500
    _v = _rng.uniform(-3.0, 3.0, _n)
    _a = _rng.uniform(0.3, 2.5, _n)
    _z = _rng.uniform(0.1, 0.9, _n)
    _t = _rng.uniform(0.0, 1.0, _n)
    _obs = np.column_stack([_t + 10 ** _rng.uniform(-3.0, 1.0, _n), np.sign(_v)])

    _obs_var = pt.matrix("obs", dtype="float64")
    _param_vars = [pt.vector(name, dtype="float64") for name in "vazt"]
    _pytensor_logp = pytensor.function(
        [_obs_var, *_param_vars], logp_ddm(_obs_var, *_param_vars)
    )
    _difference = np.abs(
        np.asarray(logp_ddm_jax(_obs, _v, _a, _z, _t))
        - _pytensor_logp(_obs, _v, _a, _z, _t)
    )
    mo.md(f"Largest absolute difference over {_n} trials: `{_difference.max():.1e}`")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Simulate a two-armed bandit

    `ssms.rl` simulates the same preset, so the data use the columns and response
    coding the model expects: `participant_id`, `rt`, `response` (`-1` or `1`)
    and `feedback`, grouped by participant in trial order, with the same number of
    trials per participant.

    `ssms` simulates the DDM with Euler steps of 1 ms by default. That widens the
    effective boundary slightly, and an exact likelihood then recovers `a` about
    0.02 above its generating value. A step of 0.1 ms shrinks that offset to
    about 0.005–0.01, well inside the posterior uncertainty, so we set it through
    the preset's `ssm_kwargs`.
    """)
    return


@app.cell
def _(rl):
    true_params = {"rl_alpha": 0.3, "scaler": 3.0, "a": 1.2, "z": 0.5, "t": 0.3}
    preset = rl.preset.get("2AB_RW_DDM")
    preset.ssm_kwargs = {"delta_t": 1e-4, "max_t": 20.0}
    simulator = rl.Simulator(preset)
    data = simulator.simulate(
        theta=true_params, n_trials=200, n_participants=5, random_state=1
    )
    data.head()
    return data, simulator, true_params


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fit with the analytical and the LAN likelihood

    The two models differ only in `decision_process_loglik_kind`. The simulated
    data contain no lapses, so both switch off HSSM's default 5% lapse component
    with `p_outlier=None`; with real data, keep it unless you have a reason not
    to. Both use numpyro, the default sampler for RLSSM models.
    """)
    return


@app.cell
def _(FULL_RUN, data, hssm, time):
    _draws = 1_000 if FULL_RUN else 300
    fits = {}
    for _kind in ["analytical", "approx_differentiable"]:
        _model = hssm.rl.RLSSM(
            data,
            model="2AB_RW_DDM",
            decision_process_loglik_kind=_kind,
            p_outlier=None,
        )
        _start = time.perf_counter()
        _idata = _model.sample(
            chains=4, draws=_draws, tune=_draws, random_seed=1, progressbar=False
        )
        fits[_kind] = {"idata": _idata, "seconds": time.perf_counter() - _start}
    return (fits,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Compare the posteriors
    """)
    return


@app.cell
def _(az, fits, mo, true_params):
    _rows = []
    for _name, _value in true_params.items():
        _row = f"| `{_name}` | {_value} |"
        for _kind in ["analytical", "approx_differentiable"]:
            _summary = az.summary(
                fits[_kind]["idata"],
                var_names=[_name],
                ci_kind="hdi",
                ci_prob=0.94,
                round_to="none",
            ).iloc[0]
            _row += (
                f" {_summary['mean']:.3f} [{_summary['hdi94_lb']:.3f}, "
                f"{_summary['hdi94_ub']:.3f}] | {_summary['r_hat']:.3f} |"
            )
        _rows.append(_row)

    _stats = []
    for _kind in ["analytical", "approx_differentiable"]:
        _divergences = int(fits[_kind]["idata"]["sample_stats"]["diverging"].sum())
        _stats.append(
            f"`{_kind}`: {fits[_kind]['seconds']:.0f} s, {_divergences} divergences"
        )

    mo.md(
        "| parameter | truth | analytical: mean [94% HDI] | r_hat "
        "| LAN: mean [94% HDI] | r_hat |\n|---|---|---|---|---|---|\n"
        + "\n".join(_rows)
        + "\n\nSampling time on the machine that built this page, and divergences: "
        + "; ".join(_stats)
        + "."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Both likelihoods describe the same data, so their posteriors should be close.
    The analytical likelihood is exact and usually samples faster. It also has no
    training range. The learner's drift `v = scaler * (Q1 - Q0)` can reach ±10
    within the preset's `scaler` bounds, while the DDM network was trained on `v`
    in (-3, 3) and nothing checks the computed drift against that range.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the fit by simulation

    `model.sample_posterior_predictive()` does not support RLSSM models yet,
    whichever likelihood they use. Simulate from `ssms.rl` instead: with
    `mode="ppc"`, the simulator replays the observed task, so the learner sees the
    same feedback each participant did. Here we simulate once at the analytical
    posterior means. This limitation is also why HSSM warned, when it built each
    model above, that `2AB_RW_DDM` is not supported in `ssm_simulators`: the
    warning concerns predictive sampling only, not the fit.
    """)
    return


@app.cell
def _(data, fits, mo, simulator, true_params):
    _posterior = fits["analytical"]["idata"]["posterior"]
    _means = {name: float(_posterior[name].mean()) for name in true_params}
    _predicted = simulator.simulate(
        theta=_means, mode="ppc", observed_data=data, random_state=2
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

    - **Which decision processes.** `"analytical"` works for decision processes
      whose analytical likelihood declares a JAX implementation (`jax_loglik` in
      its model configuration): the DDM, and the LBA with 2, 3 or 4 choices
      (`lba2`, `lba3`, `lba4`; see below). Asking for it on a preset without one,
      such as `"2AB_RW_Angle"`, raises an error that names what is missing.
    - **Other entry points.** `hssm.rl.get_rlssm_model_config` and
      `hssm.rl.RLSSMConfig.from_ssms_model` take the same
      `decision_process_loglik_kind`. A model registered with
      `hssm.rl.register_rlssm_model(..., decision_process="ddm",
      decision_process_loglik_kind="analytical")` takes its decision-parameter
      bounds from the analytical likelihood; the presets keep their `ssms` bounds
      whichever likelihood you choose.
    - **Passing a config.** With `model_config=`, the config's
      `decision_process_loglik_kind` decides. Passing a different
      `decision_process_loglik_kind` alongside it raises an error rather than
      silently using one of the two.
    - **Response coding.** Keep the `-1`/`1` coding of the simulated data: the
      preset's learner maps it to the two arms itself.
    - **Sampler.** Both kinds default to numpyro. If you choose
      `sampler="pymc"`, use `cores=1` (or `mp_ctx="forkserver"`) on Apple-silicon
      Macs: PyMC's default process start method can hang with HSSM's JAX
      likelihoods there.

    ### An LBA decision process

    No `ssms` preset pairs a learner with the LBA yet, so build the model as an
    `ssms.rl.ModelConfig`. `RescorlaWagnerRaceDrifts` gives arm `i` the drift
    `v_i = scaler * q_i`, and responses are coded `0, ..., n - 1`:

    ```python
    from ssms import rl
    from ssms.rl.env import Bandit
    from ssms.rl.learning import RescorlaWagnerRaceDrifts

    rw_lba3 = rl.ModelConfig(
        model_name="3AB_RW_LBA3",
        description="Three-armed bandit, Rescorla-Wagner learner, LBA decision process.",
        decision_process="lba3",
        learning_process=RescorlaWagnerRaceDrifts(n_actions=3, initial_q=0.5),
        task_environment=Bandit.bernoulli(
            probabilities=[0.7, 0.2, 0.1], response_labels=[0, 1, 2]
        ),
    )
    lba_data = rl.Simulator(rw_lba3).simulate(
        theta={"rl_alpha": 0.3, "scaler": 3.0, "A": 0.3, "b": 0.6},
        n_trials=200,
        n_participants=5,
    )
    config = hssm.rl.RLSSMConfig.from_ssms_model(
        rw_lba3, decision_process_loglik_kind="analytical"
    )
    model = hssm.rl.RLSSM(lba_data, model_config=config, p_outlier=None)
    ```

    The free parameters are `rl_alpha`, `scaler`, `A` and `b`; this LBA has no
    non-decision time. Its simulator ignores `random_state` (ssm-simulators#399),
    so seed `np.random` for reproducible data.
    """)
    return


if __name__ == "__main__":
    app.run()
