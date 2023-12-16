# Changelog

## 0.2.x

### 0.2.0

This is a major version update! Many changes have taken place in this version:

#### Breaking changes

When `hierarchical` argument of `hssm.HSSM` is set to `True`, HSSM will look into the
`data` provided for the `participant_id` field. If it does not exist, an error will
be thrown.

### New features

- Added `link_settings` and `prior_settings` arguments to `hssm.HSSM`, which allows HSSM
  to use intelligent default priors and link functions for complex hierarchical models.

- Added an `hssm.plotting` submodule with `plot_posterior_predictive()` and
  `plot_quantile_probability` for creating posterior predictive plots and quantile
  probability plots.

- Added an `extra_fields` argument to `hssm.HSSM` to pass additional data to the
  likelihood function computation.

- Limited `PyMC`, `pytensor`, `numpy`, and `jax` dependency versions for compatibility.

## 0.1.x

### 0.1.5

We fixed the errors in v0.1.4. Sorry for the convenience! If you have accidentally
downloaded v0.1.4, please make sure that you update hssm to the current version.

- We made Cython dependencies of this package available via pypi. We have also built
  wheels for (almost) all platforms so there is no need to build these Cython
  dependencies.

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
