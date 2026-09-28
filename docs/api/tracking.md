`hssm.track` records an inference workflow — `HSSM(...)`, `sample()` or `vi()`,
and `save_model()` — as one MLflow run: the model and its priors, the sampler
settings, convergence diagnostics, the traces, and any parameters, metrics or
figures you add yourself. Where the likelihood is a published network, the run
also carries that network's provenance. Tracking is opt-in and requires the
optional `tracking` extra (`pip install hssm[tracking]`). See
[Track fits with MLflow](../how_to/track_with_mlflow.md) for usage.

:::hssm.tracking
