# Track fits with MLflow

HSSM can record each fit to an [MLflow](https://mlflow.org) tracking server: the
model and network used, sampler settings, convergence diagnostics, and the traces.
Combined with the records ssm-simulators and LANfactory write for data generation
and network training, one `lineage_id` follows a dataset through training to every
fit that used the resulting network.

Tracking is **off by default** and needs the optional dependency:

```bash
pip install "hssm[tracking]"
```

## Basic use

```python
import hssm

with hssm.track(experiment="infer/ddm", run_name="pilot-01"):
    model = hssm.HSSM(data, model="ddm")
    model.sample(draws=1000, chains=4)
    model.save_model()
```

Everything inside the block goes to one MLflow run. `sample()` logs the sampler
settings and convergence metrics; `save_model()` can attach the pickled model.
Runs land wherever `MLFLOW_TRACKING_URI` points (your lab's shared server, if you
have one), or pass `tracking_uri=` explicitly.

## What is recorded

| Kind | Keys |
| --- | --- |
| Params | `model`, `loglik_kind`, `network_file`, `hf_revision`, `sampler`, `draws`, `tune`, `chains`, `target_accept`, `n_trials`, `n_subjects`, `spec_sha256`, `hssm_version`, `pymc_version`, `bambi_version` |
| Tags | `schema_version="2"`, `phase="infer"`, `lineage_id`, `lineage_source`, `mlflow_run_id_train`, `data_sha256`, `user`, `hostname`, `git_sha`, SLURM ids |
| Metrics | `sampling_seconds`, `divergences`, `r_hat_max`, `ess_bulk_min`, `ess_tail_min` |
| Artifacts | `model_spec.json`, `summary.csv`, `traces.nc`; `model.pkl` with `log_artifacts="all"` |

`spec_sha256` hashes the modelling choices (model, `include`, priors, links,
`p_outlier`, …) but not the data, so two fits share it exactly when the model
specification is the same. `data_sha256` hashes the observed data frame.

## Lineage

When the likelihood is a network downloaded from `franklab/HSSM`, HSSM reads the
repository's `manifest.json` (maintained by LANfactory's `upload-hf`) and copies
that network's `lineage_id` and training run id onto the fit. Fits of analytical
likelihoods, or of networks not in the manifest, get a freshly minted id; the
`lineage_source` tag says which happened (`hf_manifest`, `user`, or `minted`).
Pass `lineage_id=` to `track()` to override.

## Extras

```python
with hssm.track(experiment="infer/ddm", log_artifacts="all") as run:
    model = hssm.HSSM(data, model="ddm")
    model.sample()
    run.log_figure(model.plot_trace()[0, 0].figure, "plots/trace.png")
    run.log_metric("loo_elpd", float(az.loo(model.traces).elpd_loo))
```

`log_artifacts=False` records params and metrics only, which is useful when traces
are large and already saved elsewhere.

## Querying

```python
import mlflow
mlflow.set_tracking_uri("http://<mlflow-host>:5000")

# every fit built on one dataset lineage
mlflow.search_runs(search_all_experiments=True,
                   filter_string="tags.lineage_id = '<id>'")

# poorly converged ddm fits
mlflow.search_runs(search_all_experiments=True,
                   filter_string="tags.phase = 'infer' AND params.model = 'ddm' "
                                 "AND metrics.r_hat_max > 1.01")
```

The full schema shared by all three packages lives in the HSSMSpine repository at
`_docs/mlflow-schema.md`.
