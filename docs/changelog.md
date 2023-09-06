# Changelog

## 0.1.x

### 0.1.4

- Added support of `blackbox` likelihoods for `ddm` and `ddm_sdv` models.
- Added support for `full_ddm` models via `blackbox` likelihoods.
- Added the ability to use `hssm.Param` and `hssm.Prior` to specify model parameters.
- Added support for non-parameter fields to be involved in the computation of likelihoods.
- Major refactor of the code to improve readability and maintainability.
- Fixed a bug in model.sample_posterior_predictive().

### 0.1.3

- Added the ability to specify `inf`s in bounds.
- Fixed an issue where `nuts_numpyro` sampler fails with regression and lapse distribution.
- Defaults to `nuts_numpyro` sampler with `approx_differentiable` likelihoods and `jax` backend.
- Added a `hssm.show_defaults()` convenience function to print out default configs.
- Added default `blackbox` likelihoods for `ddm` and `ddm_sdv` models.
- Various under-the-hood documentation improvements.

### 0.1.2

- Improved numerical stability of `analytical` likelihoods.
- Added the ability to handle lapse distributions.
- Added the ability to perform prior predictive sampling.
- Improved model information output - now default priors provided by `bambi` is also printed.
- Added a `hierarchical` switch which turns all parameters into hierarchical
  when `participant_id` is provided in data.
- Parameters are now named more consistently (group-specific terms are now aliased correctly).

- Fixed a bug where information about which parameter is regression is incorrectly passed.
- Added links to Colab to try out hssm in Google Colab.

### 0.1.1

- Handle `float` types in `PyTensor` and `JAX` more consistently and explicitly.
- Updated model output format to include likelihood kinds and display bounds more consistently.
- Support for `inf`s in bounds.
- Convenient method for simulating data with `ssm_simulators`.
- More test coverage.
- CI workflows for publishing package to PyPI.
- Enhancement to documentations.
