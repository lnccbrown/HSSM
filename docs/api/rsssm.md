Use the `hssm.RSSSM` class to fit a **regime-switching sequential sampling
model**: each trial belongs to one of `K` hidden regimes that evolve as a Markov
chain, and within a regime the `(rt, response)` emission is a standard SSM (e.g.
DDM) with regime-specific *switching* parameters. The discrete regimes are
marginalised out by the forward algorithm, so only continuous parameters remain
for NUTS. After sampling, `infer_regimes` reconstructs the latent regime
trajectories (Forward-Filter Backward-Sample) and `compute_log_likelihood`
produces the per-trial log-likelihood needed for `arviz.loo`.

::: hssm.RSSSM
    handler: python
    options:
        show_root_heading: true
        show_signature_annotations: true
        show_object_full_path: false
        show_signature: true  # Make sure this is true
        docstring_options:
            ignore_init_summary: false
        members:
            - traces
            - pymc_model
            - sample
            - infer_regimes
            - compute_log_likelihood
            - plot_regime_recovery
            - find_MAP
            - graph
