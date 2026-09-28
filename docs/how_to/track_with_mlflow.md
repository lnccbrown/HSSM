# Track fits with MLflow

Fitting the same data several ways is how most analyses go, and by the fourth
model it is easy to lose track of which settings produced which result. HSSM can
record each fit to [MLflow](https://mlflow.org): the model and its priors, the
sampler settings, convergence diagnostics, the traces, and whatever else you
want to note about the run.

Tracking is **off by default** and needs the optional dependency:

```bash
pip install "hssm[tracking]"
```

## Record a fit

Wrap the work in `hssm.track`:

```python
import hssm

data = hssm.load_data("cavanagh_theta")

with hssm.track(experiment="my-study", run_name="ddm-basic"):
    model = hssm.HSSM(data, model="ddm")
    model.sample(draws=1000, chains=4)
```

Everything inside the block becomes one MLflow run. With nothing else
configured, two things appear in your working directory: `mlflow.db` holds the
record of each run, and `mlruns/` holds the files attached to it — the traces,
the summary table, the model specification. Keep both; the database refers to
the artifacts by path rather than storing them.

Browse them with:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

To keep several projects apart, or to use a tracking server, set
`MLFLOW_TRACKING_URI` in your environment or pass `tracking_uri=` to `track()`.

## Record what you varied

`params=` is for the dimensions *you* are changing — the things that make one
run different from the next. They become sortable columns in the MLflow UI, so a
study stays readable:

```python
for sd in [0.5, 1.0, 2.0]:
    with hssm.track(experiment="my-study", params={"prior_v_sd": sd}):
        model = hssm.HSSM(
            data,
            model="ddm",
            v={"prior": {"name": "Normal", "mu": 0.0, "sigma": sd}},
        )
        model.sample()
```

Add more as you go with `run.log_param(...)`, and attach figures or your own
metrics the same way:

```python
import arviz as az

with hssm.track(experiment="my-study") as run:
    model = hssm.HSSM(data, model="ddm")
    model.sample()
    run.log_param("note", "pilot subjects only")
    run.log_metric("loo_elpd", float(az.loo(model.traces).elpd))
    trace_plot = az.plot_trace_dist(model.traces, var_names=["v"])
    run.log_figure(trace_plot.viz["figure"].item(), "plots/trace.png")
```

Keys HSSM records itself are refused rather than silently dropped — MLflow
rejects a whole batch of parameters when one key repeats with a different value,
which would cost you the rest of the run's record.

## What gets recorded

| Kind | Keys |
| --- | --- |
| Params | `model`, `loglik_kind`, `network_file`, `hf_revision`, `sampler`, `draws`, `tune`, `chains`, `target_accept`, `n_trials`, `n_subjects`, `spec_sha256`, `hssm_version`, `pymc_version`, `bambi_version` |
| Metrics | `sampling_seconds`, `divergences`, `r_hat_max`, `ess_bulk_min`, `ess_tail_min` |
| Tags | `user`, `hostname`, `git_sha`, `data_sha256`, `schema_version`, `phase`, `lineage_id`, `lineage_source` |
| Artifacts | `model_spec.json`, `summary.csv`, `traces.nc`; `model.pkl` with `log_artifacts="all"` |

`spec_sha256` hashes the modelling choices — model, `include`, priors, links,
`p_outlier` — but not the data, so two fits share it exactly when the
specification is the same. `data_sha256` hashes the observed data frame and its
schema, so fits of the same data share that instead. Between them, the runs of
one study group themselves without you naming anything.

`log_artifacts=False` records params and metrics only, which is worth using when
traces are large and already saved elsewhere.

## Variational inference

`vi()` is tracked the same way, recording `method`, `niter`, `draws` and
`backend`, the `vi_seconds` it took, the final and best ELBO, and the
approximate posterior as `vi_traces.nc`:

```python
with hssm.track(experiment="my-study", run_name="advi"):
    model = hssm.HSSM(data, model="ddm")
    model.vi(niter=20000)
```

## Find runs again

```python
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# every fit of one dataset
mlflow.search_runs(search_all_experiments=True,
                   filter_string="tags.data_sha256 = '<hash>'")

# the ones that did not converge
mlflow.search_runs(experiment_names=["my-study"],
                   filter_string="metrics.r_hat_max > 1.01")
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

This matters when a network is being brought into production and nobody else
needs to think about it. The schema shared across ssm-simulators, LANfactory and
HSSM lives in the HSSMSpine repository at `_docs/mlflow-schema.md`.

## See also

- [Compare and interpret models](compare_models.ipynb)
- [Run variational inference](../tutorials/variational_inference.ipynb)
- [Save and load fitted models](../tutorials/save_load_tutorial.ipynb)
