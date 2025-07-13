# Changelog

### 0.2.7

This version of HSSM incorporated the following changes:

1. Bugfix in `mkdocs.yml`. No tutorial notebooks should be actively executed when re-building docs.

### 0.2.6

This version of HSSM incorporated the following changes:

1. Overhaul of our development infrastructure to depend on `uv`
2. Addition of various new tutorials (highlight: how to fit RL+SSM models)
3. New `rldm` likelihoods
4. Major internal refactor (e.g. `DataValidator` class for pre-processing)

### 0.2.5

This version of HSSM incorporated the following changes:

1. We added a new tutorial on how to use custom likelihood functions with HSSM.
2. Added convenience function `compile_logp()` to return a compiled log-likelihood function to be used freely downstream.
3. Some improvements to internals to allow for (1.)
4. Major improvments to plotting with `model_cartoon_plots()`
5. Refactoring and minor improvements to pre-existing plots
6. Added functionality to save and load models with `save_model()` and `load_model()`.

### 0.2.4

This version of HSSM incorporated the following changes:

1. We updated HSSM to be compatible with the major API changes in `bambi` v0.14.0.
2. We fixed various graphing issues in `pymc` 5.16.0+, thanks to the API changes in `bambi`.
3. We added variational inference via native `pymc`.
4. We can now use `float64` inference.
5. We fixed some minor bugs in providing initial values.
6. We added a model.dic() convenience function.
7. We added a model.restore_traces() convenience function.
8. Other minor bug fixes.

### 0.2.3

This is a maintenance release of HSSM, mainly to add a version constraint on `bambi` in light of the many breaking changes that version `0.1.4` introduces. This version also improved compatibility with `PyMC>=5.15` and incorporated minor bug fixes:

1. We incorporated a temporary fix to graphing which broke after `PyMC>=5.15`.
2. We deprecated `ndim` and `ndim_supp` definition in `SSMRandomVariable` in `PyMC>-5.16`.
3. We fixed a bug that prevents new traces from being returned if `model.sample()` is called again.

### 0.2.2

HSSM is now on Conda! We now recommend installing HSSM through `conda install -c conda-forge hssm`. For advanced users, we also support installing the GPU version of JAX through `pip install hssm[cuda12]`.

This version incorporates various bug fixes:

1. We fixed a major bug that causes divergences for models using `approx_differentiable` and `blackbox` likelihoods. We are still looking into the issues of divergence with `analytical` likelihoods.
2. We made the model creation process more robust, fixing errors when categorical variables are used with group identifiers.
3. We updated the codebase according to the deprecations in higher versions of JAX.
4. We implemented a temporary fix to an issue that might cause the kernel to die due to OOM.

### 0.2.1

We added a few new features in 0.2.1:

1. We have finished updating the HSSM code base to support go-nogo data and deadline. We will provide documentation once the networks are added to our huggingface repo.
2. We updated `hssm.distribution_utils` to streamline the creation of `pm.Distribution`s.
3. We now support response variables other than `rt` and `response`. They can be specified through `model_config` via the new `response` field.
4. We have fixed some of the issues with convergence when using `log-logit` link functions and/or safe priors.

Other minor updates

- Fixed an incompatible shape error during posterior predictive sampling when `p_outlier` is estimated as a parameter.
- Updated documentation for using `make_distribution` with PyMC.

Bug fixes:

- Fixed default list of parameters for `ddm_full` model and the bounds for `ddm_sdv` model.

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
