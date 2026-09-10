# Reference

This section records the public HSSM interfaces and project metadata. Use the
[learning index](../learn/index.md) for guided examples or the
[how-to index](../how_to/index.md) for task-oriented procedures.

## Core modeling interfaces

- [`hssm.HSSM`](../api/hssm.md) builds and fits an HSSM model.
- [`hssm.aDDM` and `hssm.aDDMConfig`](../api/addm.md) provide the specialized
  attentional drift-diffusion interface.
- [`hssm.Param`](../api/param.md), [`hssm.Prior`](../api/prior.md), and
  [`hssm.Link`](../api/link.md) describe parameter formulas, priors, and links.
- [`hssm.ModelConfig`](../api/model_config.md) describes registered model
  configurations.
- [`hssm.rl`](../api/rl.md) contains the reinforcement-learning interfaces.
- [Built-in models and likelihoods](models-and-likelihoods.md) records every
  `hssm.HSSM(model=...)` name, configured likelihood kind, parameter, and choice.
- [The ONNX likelihood contract](../how_to/custom_onnx_likelihoods.md) records
  the portable artifact boundary for approximate differentiable likelihoods.

## Functions and specialized modules

- [`hssm.list_data`](../api/list_data.md) lists package-supplied datasets.
- [`hssm.load_data`](../api/load_data.md) loads package-supplied datasets.
- [`hssm.simulate_data`](../api/simulate_data.md) simulates supported models.
- [`hssm.show_defaults`](../api/show_defaults.md) inspects model defaults.
- [`hssm.set_floatX`](../api/set_floatx.md) selects floating-point precision.
- [`hssm.check_data_for_rl`](../api/check_data_for_rl.md) validates RLSSM data.
- [`hssm.list_models` and `hssm.register_model`](../api/model_registry.md)
  discover and extend the HSSM model registry.
- [`hssm.likelihoods`](../api/likelihoods.md) records likelihood interfaces.
- [`hssm.distribution_utils`](../api/distribution_utils.md) records distribution
  helpers.
- [`hssm.plotting`](../api/plotting.md) records plotting functions.

## Project reference

- [Changelog](../changelog.md) — release-by-release behavior changes.
- [Ecosystem map](../ecosystem/index.md) — package ownership and contributor
  routes across the native HSSM ecosystem.
- [Contribution guide](../CONTRIBUTING.md) and
  [local development setup](../local_development.md).
- [Credits](../credits.md) and [license](../license.md).

## Historical tutorial snapshots

These archives preserve the material presented at a specific event. Prefer the
current Learn and How-to pages above for maintained guidance.

- [Winterbrain 2025 — Talk 1](../archive/hssm_tutorial_workshop_1.ipynb).
- [Winterbrain 2025 — Talk 2](../archive/hssm_tutorial_workshop_2.ipynb).
- [PyMC to HSSM — MathPsych 2025](../archive/pymc_to_hssm.ipynb).
