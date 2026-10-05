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
    dataset_name="cavanagh_theta",  # optional but recommended for dataset tracking
):
    model = hssm.HSSM(data, model="ddm")
    model.sample(draws=1000, chains=4)
```

Everything inside the block becomes one MLflow run. With nothing else configured, two things appear in your working directory: `mlflow.db` holds the record of each run, and `mlruns/` holds the files attached to it — the traces, the summary table, the model specification. Keep both; the database refers to the artifacts by path rather than storing them.

Browse them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`dataset_name=` is optional but worth setting: it fills the run's **Datasets** panel with your data's column schema and row count, and lets the UI filter by dataset. Left out, that panel stays empty — a DataFrame carries no name of its own, and an invented one would look identical for every study. Either way the data is identified by the `data_sha256` tag.

One block records one fit, and fitting more than once inside a single block is
not supported: only the first fit is recorded, and a later one is skipped with
a warning.
This applies to a second model, to re-sampling the same model, and to `vi()`
after `sample()`.

To keep several projects apart, or to use a tracking server, set `MLFLOW_TRACKING_URI` in your environment or pass `tracking_uri=` to `track()`.

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

## Reserved keys

Keys HSSM records itself are refused rather than silently dropped — MLflow rejects a whole batch of parameters when one key repeats with a different value, which would cost you the rest of the run's record.

## What gets recorded

| Kind | Keys |
| --- | --- |
| Params | `model`, `loglik_kind`, `network_file`, `hf_revision`, `missing_data_network_file`, `missing_data_hf_revision`, `sampler`, `draws`, `tune`, `chains`, `target_accept`, `n_trials`, `n_subjects`, `spec_sha256`, `hssm_version`, `pymc_version`, `bambi_version` |
| Metrics | `sampling_seconds`, `divergences`, `r_hat_max`, `ess_bulk_min`, `ess_tail_min` |
| Tags | `user`, `hostname`, `git_sha`, `data_sha256`, `schema_version`, `phase`, `lineage_id`, `lineage_source` |
| Artifacts | `model_spec.json`, `summary.csv`, `traces.nc`; `model.pkl` with `log_artifacts="all"` |

`spec_sha256` hashes the modelling choices — model, `include`, priors, links, `p_outlier` — but not the data, so two fits share it exactly when the specification is the same. `data_sha256` hashes the observed data frame and its schema, so fits of the same data share that instead. Together they separate "same model, different data" from "same data, different model".

`log_artifacts=False` records params and metrics only, for when traces are
large and already stored elsewhere.

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

`search_runs` returns a DataFrame, so a study's results can go straight into a
table or a plot.

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
