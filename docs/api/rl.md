# hssm.rl

The `hssm.rl` module provides reinforcement-learning extensions for HSSM,
integrating reinforcement-learning update rules with sequential-sampling
decision models (SSMs).

## RLSSM

Use the `hssm.rl.RLSSM` class to construct a reinforcement-learning sequential
sampling model from a named model string.

::: hssm.rl.RLSSM
    handler: python
    options:
        show_root_heading: true
        show_signature_annotations: true
        show_object_full_path: false
        show_signature: true  # Make sure this is true
        docstring_options:
            ignore_init_summary: false
        members:
            - list_models
            - sample
            - sample_posterior_predictive
            - sample_prior_predictive
            - vi
            - find_MAP
            - log_likelihood
            - plot_predictive
            - plot_quantile_probability
            - graph
            - pymc_model

## RLSSMConfig

Configuration object for RLSSM models.

::: hssm.rl.RLSSMConfig
    handler: python
    options:
        show_root_heading: true
        show_signature_annotations: true
        show_object_full_path: false
        show_signature: true

## Registry functions

Helpers for discovering, building, and registering named RLSSM models and
custom SSM base log-likelihood functions.

::: hssm.rl.get_rlssm_model_config
    handler: python
    options:
        show_root_heading: true
        show_signature_annotations: true
        show_object_full_path: false
        show_signature: true

::: hssm.rl.list_models
    handler: python
    options:
        show_root_heading: true
        show_signature_annotations: true
        show_object_full_path: false
        show_signature: true

::: hssm.rl.register_rlssm_model
    handler: python
    options:
        show_root_heading: true
        show_signature_annotations: true
        show_object_full_path: false
        show_signature: true

::: hssm.rl.register_ssm
    handler: python
    options:
        show_root_heading: true
        show_signature_annotations: true
        show_object_full_path: false
        show_signature: true

## Utilities

::: hssm.rl.validate_balanced_panel
    handler: python
    options:
        show_root_heading: true
        show_signature_annotations: true
        show_object_full_path: false
        show_signature: true
