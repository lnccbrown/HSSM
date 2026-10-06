# Track fits with MLflow

HSSM can record each fit to [MLflow](https://mlflow.org): the model and its
priors, the sampler settings, convergence diagnostics, the traces, and whatever
else you want to note about the run.

Tracking is **off by default** and needs the optional dependency:

```bash
uv add "hssm[tracking]"       # or: uv pip install "hssm[tracking]"
pip install "hssm[tracking]"  # with pip
```

## Record a fit

Wrap the work in `hssm.track`:

```python
import hssm

data = hssm.load_data("cavanagh_theta")

with hssm.track(
    experiment="my-study",
    run_name="ddm-basic",
    dataset_name="cavanagh_theta",
):
    model = hssm.HSSM(data, model="ddm")
    model.sample(draws=1000, chains=4)
```

Everything inside the block becomes one MLflow run. With nothing else configured, two things appear in your working directory:
- `mlflow.db` holds the record of each run, and
- `mlruns/` holds the files attached to it: the traces, the summary table, the model specification, and a copy of the data the model was fit to (`data.parquet`).

The database refers to the artifacts by path rather than storing them.

Browse them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`dataset_name=` is optional but worth setting: it fills the run's **Datasets** panel with your data's column schema and row count, and lets the UI filter by dataset. Left out, that panel stays empty — a DataFrame carries no name of its own, and an invented one would look identical for every study. Either way the data is identified by the `data_sha256` tag, and the data itself is stored with the run as `data.parquet` (unless `log_artifacts=False`); with a name, the Datasets entry links to that stored copy.

One block records one fit, and fitting more than once inside a single block is
not supported: only the first fit is recorded, and a later one is skipped with
a warning.
This applies to a second model, to re-sampling the same model, and to `vi()`
after `sample()`.

To keep several projects apart, or to use a tracking server, set `MLFLOW_TRACKING_URI` in your environment or pass `tracking_uri=` to `track()`; see [Use a shared store](#use-a-shared-store).

## Recording parameter changes

`params=` records your own run-level variables. MLflow pivots its run comparison on params, so whatever you are varying belongs here rather than in tags:

```python
for sd in [0.5, 1.0, 2.0]:
    with hssm.track(
        experiment="my-study",
        params={"prior_v_sd": sd},
        dataset_name="cavanagh_theta",
    ):
        model = hssm.HSSM(
            data,
            model="ddm",
            v={"prior": {"name": "Normal", "mu": 0.0, "sigma": sd}},
        )
        model.sample()
```

Add more as you go with `run.log_param(...)`, and attach figures or your own metrics the same way:

```python
import arviz as az

with hssm.track(experiment="my-study", dataset_name="cavanagh_theta") as run:
    model = hssm.HSSM(data, model="ddm")
    model.sample()
    run.log_param("note", "pilot subjects only")
    run.log_metric("loo_elpd", float(az.loo(model.traces).elpd))
    trace_plot = az.plot_trace_dist(model.traces, var_names=["v"])
    run.log_figure(trace_plot.viz["figure"].item(), "plots/trace.png")
```

Labels you want to filter runs by, rather than compare on, go in `tags=`, e.g.
`hssm.track(experiment="my-study", tags={"subjects": "pilot"})`.

`track()` opens an ordinary MLflow run, so anything MLflow can log works inside
the block too. Attach a file or text with MLflow's own functions:

```python
import mlflow

with hssm.track(experiment="my-study") as run:
    ...
    mlflow.log_artifact("notes.md")
    mlflow.log_dict({"exclusions": [3, 17]}, "config/exclusions.json")
```

## Reserved keys

Keys HSSM records itself are refused rather than silently dropped — MLflow rejects a whole batch of parameters when one key repeats with a different value, which would cost you the rest of the run's record.

## What gets recorded

| Kind | Keys |
| --- | --- |
| Params | `model`, `loglik_kind`, `network_file`, `hf_revision`, `missing_data_network_file`, `missing_data_hf_revision`, `sampler`, `draws`, `tune`, `chains`, `target_accept`, `cores`, `n_trials`, `n_subjects`, `spec_sha256`, `hssm_version`, `pymc_version`, `bambi_version`; for `vi()`, `method`, `niter`, `draws`, `backend` |
| Metrics | `sampling_seconds`, `divergences`, `r_hat_max`, `ess_bulk_min`, `ess_tail_min`; for `vi()`, `vi_seconds`, `elbo_final`, `elbo_max` |
| Tags | `user`, `hostname`, `hssm_version`, `data_sha256`, `schema_version`, `phase`, `lineage_id`, `lineage_source`, `model_class`, `spec_restorable`, `rebuild_artifacts`; `git_sha`, the commit of HSSM itself when it is installed from a git checkout (not of your own code); under SLURM, `slurm_job_id`, `slurm_array_job_id`, `slurm_array_task_id`; for a published network, `mlflow_run_id_train`, `run_uuid_train` |
| Artifacts | `model_spec.json`, `summary.csv`, `traces.nc` (`vi_traces.nc` for `vi()`), `data.parquet`; `model.pkl` when `save_model()` is called inside the block with `log_artifacts="all"` |

`spec_sha256` hashes the modelling choices — model, `include`, priors, links, `p_outlier` — but not the data, so two fits share it exactly when the specification is the same. `data_sha256` hashes the observed data frame and its schema, so fits of the same data share that instead. Together they separate "same model, different data" from "same data, different model".

`data.parquet` is the data the model was fit to, covariates included — the
traces keep only the observed response, so this is what lets a run show which
predictors a regression used. It is copied into every run, which matters when
runs go to a shared tracking server.

`log_artifacts=False` records params, metrics and tags only, for when traces
are large and already stored elsewhere, or when the data should not leave your
machine. A named dataset is still registered, which sends its column schema and
row count to the server but none of its values.

## Variational inference

`vi()` is tracked the same way, recording `method`, `niter`, `draws` and
`backend`, the `vi_seconds` it took, the final and best ELBO (`elbo_final`,
`elbo_max`), and the
approximate posterior as `vi_traces.nc`:

```python
with hssm.track(
    experiment="my-study",
    run_name="advi",
    dataset_name="cavanagh_theta",
):
    model = hssm.HSSM(data, model="ddm")
    model.vi(niter=20000)
```

## More than one dataset

Each run records which data it used, so fits across datasets stay separable.
Adding a simulated dataset to the same study:

```python
sim = hssm.simulate_data(model="ddm", theta=dict(v=0.5, a=1.5, z=0.5, t=0.1), size=500)

with hssm.track(
    experiment="my-study",
    run_name="ddm-recovery",
    dataset_name="ddm-sim-v0.5",
):
    model = hssm.HSSM(sim, model="ddm")
    model.sample()
```

Both names appear in the MLflow UI and can be filtered on.

## Find runs again

There are two ways to find a run. `hssm.list_runs()` gives an overview of every
run `hssm.track` recorded, one row each, newest first: id, name, experiment,
start time, status, user, model, likelihood kind, dataset, trial count, and
whether `load_run` can rebuild it (`restorable`). It takes no filters.

```python
runs = hssm.list_runs()
runs[runs["model"] == "ddm"]
```

To search on anything a run recorded, including your own params, tags and
metrics, use MLflow's `search_runs`:

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# every fit of one dataset, by the name you gave it
mlflow.search_runs(
    search_all_experiments=True, filter_string="dataset.name = 'cavanagh_theta'"
)

# or by content, which catches the same data under a different name
mlflow.search_runs(
    search_all_experiments=True, filter_string="tags.data_sha256 = '<hash>'"
)

# the ones that did not converge
mlflow.search_runs(
    experiment_names=["my-study"], filter_string="metrics.r_hat_max > 1.01"
)
```

Both return a DataFrame with a `run_id` column, so a study's results can go
straight into a table or a plot. For the run you have just fitted, the id is
on the tracker, and stays readable after the block closes:

```python
with hssm.track(experiment="my-study", run_name="ddm-basic") as run:
    model = hssm.HSSM(data, model="ddm")
    model.sample()

run_id = run.run_id
```

## Rebuild a model from a run

`hssm.load_run` turns a run back into a fitted model: it rebuilds the model from
the run's `model_spec.json` and `data.parquet`, and attaches its traces. Get the
run id as in [Find runs again](#find-runs-again).

```python
import arviz as az

model = hssm.load_run("<run_id>")
az.summary(model.traces)
```

The run must have been logged with artifacts on (the default), and the model
must have been specified with plain values: strings, numbers, formulas, and
priors written as dicts. Arguments such as `bmb.Prior` or `hssm.Link` objects
and custom likelihood functions are stored as text, so those runs cannot be
rebuilt. Only `hssm.HSSM` models are supported for now.

A likelihood network from Hugging Face is fetched at the revision the run
recorded, so the rebuilt model uses the network it was fitted with even if the
file has since been updated. A network given as a local file path is not
stored with the run: it must exist at that same path wherever the run is
rebuilt.

## Use a shared store

By default every session reads and writes `mlflow.db` in the folder Python was
started from, so a notebook started elsewhere sees none of your runs. To browse
and rebuild runs that others logged to a shared store, pass it to `list_runs`
and `load_run`:

```python
shared = (
    "sqlite:////oscar/data/<lab>/mlflow/mlflow.db"  # four slashes: an absolute path
)

hssm.list_runs(tracking_uri=shared)
model = hssm.load_run("<run_id>", tracking_uri=shared)
```

`tracking_uri=` applies to that one call; your own `track()` blocks keep logging
where they did. To use the shared store everywhere, set `MLFLOW_TRACKING_URI`
in your shell or job script instead, and `track()`, `list_runs()` and
`load_run()` all pick it up.

The database holds the list of runs, but each run's files (`data.parquet`,
`model_spec.json`, the traces) are stored at the experiment's artifact
location, an absolute path fixed when the experiment was created. `list_runs`
needs only the database; `load_run` also needs to read those files. So when
several people log to one store, create each experiment once, with an
artifact location everyone can read:

```python
import mlflow

mlflow.set_tracking_uri(shared)
mlflow.create_experiment(
    "my-study", artifact_location="/oscar/data/<lab>/mlflow/artifacts"
)
```

`create_experiment` raises if the experiment already exists; after that,
`track(experiment="my-study")` logs into it.

SQLite is fine for reading a store on a shared filesystem, but several jobs
writing to one `mlflow.db` at the same time can hit locking errors. For a store
many people log to, run a tracking server (`mlflow server --serve-artifacts`)
and use its address as `tracking_uri`: it serves the run files too, so the
artifact location stops mattering. Setting one up is described in the
HSSMSpine repository at `_docs/mlflow-deployment.md`.

## Ecosystem provenance

If your likelihood is a network published by the LAN pipeline, HSSM also records
where that network came from: it reads the repository's `manifest.json` and
copies the network's `lineage_id` and training run id onto the fit. One query
then lists every fit built on a given training dataset. Fits of analytical
likelihoods, or of networks outside the manifest, get a freshly minted id; the
`lineage_source` tag says which happened (`hf_manifest`, `user`, or `minted`).
Pass `lineage_id=` to override.

The schema shared across ssm-simulators, LANfactory and HSSM is documented in
the HSSMSpine repository at `_docs/mlflow-schema.md`.

## See also

- [Compare and interpret models](compare_models.ipynb)
- [Run variational inference](../tutorials/variational_inference.ipynb)
- [Save and load fitted models](../tutorials/save_load_tutorial.ipynb)
