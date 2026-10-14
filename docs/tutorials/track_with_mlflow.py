"""Track HSSM fits with MLflow: record runs, keep their files, rebuild a model.

This marimo notebook is the source of truth for the output-bearing Jupyter
artifact published by MkDocs. Run a quick local check with::

    uv run --group notebook --group docs marimo check --strict \
        docs/tutorials/track_with_mlflow.py
    uv run --group notebook --group docs marimo export html --no-sandbox \
        docs/tutorials/track_with_mlflow.py \
        --output /tmp/track-with-mlflow.html --force

Regenerate the committed full-run artifact with::

    FULL_RUN=1 uv run --group notebook --group docs marimo export ipynb \
        --no-sandbox docs/tutorials/track_with_mlflow.py \
        --output docs/tutorials/track_with_mlflow.ipynb \
        --include-outputs --force
    uv run ruff format docs/tutorials/track_with_mlflow.ipynb
"""

# ruff: noqa: B018, D401, E501, PLR1711  (generated marimo notebook: prose, display expressions, and bare returns)
import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import logging
    import os
    import tempfile
    import warnings

    # This tutorial does not need a GPU. Keep Molab and headless documentation
    # builds away from host CUDA plugins before importing HSSM/JAX.
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["JAX_SKIP_CUDA_CONSTRAINTS_CHECK"] = "1"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    os.environ["MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR"] = "false"
    warnings.filterwarnings("ignore")
    logging.getLogger("jax._src.xla_bridge").setLevel(logging.CRITICAL)

    # Work in a throwaway folder so the tutorial leaves nothing behind, and
    # ignore any tracking server configured in the environment.
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.chdir(tempfile.mkdtemp(prefix="hssm-mlflow-"))

    import arviz as az
    import marimo as mo
    import mlflow
    import pandas as pd

    import hssm

    # MLflow and PyMC set their log levels on import, so quiet them afterwards.
    for _name in ("alembic", "mlflow", "pymc"):
        logging.getLogger(_name).setLevel(logging.WARNING)

    FULL_RUN = os.environ.get("FULL_RUN") == "1"
    SAMPLER = {
        "draws": 1_000 if FULL_RUN else 100,
        "tune": 1_000 if FULL_RUN else 100,
        "chains": 2,
        "cores": 1,
        "random_seed": 123,
        "progressbar": False,
    }
    return SAMPLER, az, hssm, mlflow, mo, os, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Track fits with MLflow

    `hssm.track` records each fit as an [MLflow](https://mlflow.org) run. This
    notebook shows the three things that gives you:

    1. **Tracked runs**: the model, its settings and diagnostics, recorded
       without extra code, and easy to find again;
    2. **Kept artifacts**: the traces, the model specification and the data
       the model was fit to, stored with the run;
    3. **Rebuilt models**: a run turned back into a fitted model, in any
       session.

    Tracking is off by default and needs the optional extra:
    `uv add "hssm[tracking]"` or `pip install "hssm[tracking]"`.

    The setup cell above moves into a throwaway folder first. With nothing
    else configured, MLflow keeps its records in `mlflow.db` and the run files
    in `mlruns/` in your working folder; here that folder is discarded
    afterwards. The [how-to](../how_to/track_with_mlflow.md) covers using a
    shared store instead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Record a fit

    Everything inside the `hssm.track` block becomes one run. Naming the
    dataset is optional; it fills the run's **Datasets** panel in the MLflow UI.
    """)
    return


@app.cell
def _(SAMPLER, hssm, mo):
    data = hssm.load_data("cavanagh_theta")

    with hssm.track(
        experiment="my-study",
        run_name="ddm-basic",
        dataset_name="cavanagh_theta",
    ) as run:
        model = hssm.HSSM(data, model="ddm")
        model.sample(**SAMPLER)

    first_run_id = run.run_id
    mo.md(f"Recorded run `{first_run_id}`.")
    return data, first_run_id, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What a run holds

    The run records the model and its settings as **params**, the fit's
    diagnostics as **metrics**, and identifying labels as **tags**. Two keys
    group runs later: the param `spec_sha256` changes with the modelling
    choices, and the tag `data_sha256` with the data. Filter on them in
    `search_runs` as `params.spec_sha256` and `tags.data_sha256`.
    """)
    return


@app.cell
def _(first_run_id, mlflow, mo, pd):
    _run = mlflow.get_run(first_run_id)
    _params = pd.DataFrame(sorted(_run.data.params.items()), columns=["param", "value"])
    _metrics = pd.DataFrame(
        sorted((k, round(v, 3)) for k, v in _run.data.metrics.items()),
        columns=["metric", "value"],
    )
    mo.md(
        "### Params\n\n"
        + _params.to_markdown(index=False)
        + "\n\n### Metrics\n\n"
        + _metrics.to_markdown(index=False)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The run also keeps the files needed to reuse it. `data.parquet` is the
    data the model was fit to, covariates included; the traces keep only the
    observed response.
    """)
    return


@app.cell
def _(first_run_id, mlflow, mo, pd):
    _files = mlflow.MlflowClient().list_artifacts(first_run_id)
    _kept = pd.DataFrame(
        [(f.path, f"{f.file_size / 1024:,.1f} KB") for f in _files],
        columns=["file", "size"],
    )
    mo.md(_kept.to_markdown(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The files sit under the working folder, in
    `mlruns/<experiment id>/<run id>/artifacts/`. The run reports that path as
    its `artifact_uri`, and `mlflow.artifacts.download_artifacts` fetches a
    file by name:
    """)
    return


@app.cell
def _(first_run_id, mlflow, mo, os, pd):
    _uri = mlflow.get_run(first_run_id).info.artifact_uri
    _path = mlflow.artifacts.download_artifacts(
        run_id=first_run_id, artifact_path="data.parquet"
    )
    _stored = pd.read_parquet(_path)
    mo.md(
        f"Files of this run, relative to the working folder: `{os.path.relpath(_uri)}`\n\n"
        f"Its `data.parquet` holds {len(_stored):,} rows with the columns "
        + ", ".join(f"`{c}`" for c in _stored.columns)
        + "."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Vary something and compare

    `params=` records what you are varying, and `run.log_metric` adds your own
    numbers. Here two prior widths for the drift rate `v` are fit in separate
    runs, one block per fit, each logging its posterior mean of `v`.
    """)
    return


@app.cell
def _(SAMPLER, data, hssm):
    for _sd in [0.5, 2.0]:
        with hssm.track(
            experiment="my-study",
            run_name=f"ddm-v-sd-{_sd}",
            params={"prior_v_sd": _sd},
            dataset_name="cavanagh_theta",
        ) as _run:
            _model = hssm.HSSM(
                data,
                model="ddm",
                v={"prior": {"name": "Normal", "mu": 0.0, "sigma": _sd}},
            )
            _model.sample(**SAMPLER)
            _run.log_metric("v_mean", float(_model.traces.posterior["v"].mean()))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    MLflow's own `search_runs` filters on anything a run recorded, including
    your params and metrics:
    """)
    return


@app.cell
def _(mlflow, mo):
    _compared = mlflow.search_runs(
        experiment_names=["my-study"],
        filter_string="params.prior_v_sd != ''",
        order_by=["params.prior_v_sd"],
    )[
        [
            "tags.mlflow.runName",
            "params.prior_v_sd",
            "metrics.v_mean",
            "metrics.r_hat_max",
        ]
    ]
    mo.md(_compared.round(3).to_markdown(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Find runs again

    `hssm.list_runs()` lists every run `hssm.track` recorded, newest first,
    one row per run. Its `restorable` column says whether the run can be
    rebuilt. It takes no filters; filter the DataFrame instead.
    """)
    return


@app.cell
def _(hssm, mo):
    runs = hssm.list_runs()
    _ddm = runs[runs["model"] == "ddm"]
    mo.md(
        _ddm[
            [
                "run_name",
                "experiment",
                "model",
                "dataset_name",
                "n_trials",
                "restorable",
            ]
        ].to_markdown(index=False)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rebuild a model from a run

    `hssm.load_run` rebuilds the model from the run's `model_spec.json` and
    `data.parquet`, and attaches its traces. No pickle is involved, and a
    likelihood network from Hugging Face is fetched at the revision the fit
    used.
    """)
    return


@app.cell
def _(az, first_run_id, hssm, mo):
    rebuilt = hssm.load_run(first_run_id)
    mo.md(az.summary(rebuilt.traces).round(3).to_markdown())
    return (rebuilt,)


@app.cell
def _(mo, model, rebuilt):
    _same = (
        rebuilt.traces["posterior"]
        .to_dataset()
        .equals(model.traces["posterior"].to_dataset())
    )
    mo.md(f"The rebuilt model's posterior matches the original fit: **{_same}**.")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Browse the runs

    This notebook keeps its runs in a throwaway folder, so they are gone once
    it finishes. To keep and browse your own, delete the `os.chdir(...)` line
    in the first cell and run the notebook again: the runs then go to
    `mlflow.db` in the folder you started it from. From that folder, start the
    MLflow UI and open the address it prints:

    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```

    The [Track fits with MLflow](../how_to/track_with_mlflow.md) how-to lists
    everything a run records, how to attach your own files, and how to read a
    store other lab members log to.
    """)
    return


if __name__ == "__main__":
    app.run()
