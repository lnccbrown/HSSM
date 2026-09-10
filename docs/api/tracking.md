`hssm.track` records an inference workflow as one MLflow run following the HSSM
ecosystem run schema (`phase=infer`), so a fit can be traced back to the network it
used and the data that network was trained on. Tracking is opt-in and requires the
optional `tracking` extra (`pip install hssm[tracking]`). See
[Track fits with MLflow](../how_to/track_with_mlflow.md) for usage.

:::hssm.tracking
