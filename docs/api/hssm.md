Use `hssm.HSSM` class to construct an HSSM model.

::: hssm.HSSM
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
            - sample_posterior_predictive
            - sample_prior_predictive
            - vi
            - find_MAP
            - log_likelihood
            - summary
            - plot_trace
            - graph
            - plot_posterior_predictive
            - plot_quantile_probability
            - restore_traces
            - initial_point
