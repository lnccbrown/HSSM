# Changelog

### Unreleased

#### Major new features:

1. Per-parameter centered vs. non-centered parameterization. Pass `noncentered` to `HSSM(...)` as a `dict` keyed by parameter name (e.g. `noncentered={"v": False, "a": True}`), or set a per-prior `noncentered` field on a group term's prior (dict or `hssm.Prior`) to override the model-level choice. Requires Bambi >= 0.19. See the "Per-parameter centered vs. non-centered parameterization" tutorial.
2. **Attentional Drift Diffusion Model (aDDM)**: new `hssm.aDDM` model (with `aDDMConfig`) built on a vendored, differentiable JAX first-passage-time likelihood. Supports per-trial fixation covariates (`r1, r2, flag, sacc_array, d, sigma`), **trial-wise regression / hierarchical priors** on the core parameters (`eta, kappa, a, b, x0`), a sampled non-decision time `t`, and posterior-predictive checks conditioned on the observed fixations (via the aDDM simulator in ssm-simulators). See the "Attentional DDM" tutorial and `scripts/addm_parameter_recovery.py` for an end-to-end recovery check.

#### Dependency changes:

1. `bambi` floor raised to `>=0.19.0` (per-parameter non-centered support); `ssm-simulators` floor raised to `>=0.13.1` (aDDM engine + fixation continuation).

#### Bug fixes:

1. **Python 3.14: posterior/prior predictive sampling fixed.** On 3.14, the dynamically created SSM random-variable class carried PEP 649 annotation metadata that numba's vendored cloudpickle could not serialize (`TypeError: cannot pickle '_abc._abc_data' object`), breaking `sample_posterior_predictive` and related plotting under the numba backend. The class attributes are now un-annotated plain assignments, and the previous 3.14 xfail markers on the predictive/plotting tests have been removed.
2. **`model.vi(..., backend="jax")` no longer crashes inside HSSM (#1056).** PyMC 6 added a `backend` option to `pm.fit()`, which `HSSM.vi()` forwards. Requesting `backend="jax"` failed with `NotImplementedError: No JAX conversion for the given Op: LANLogpVJPOp`, because the operator that computes LAN likelihood gradients had no JAX translation. It now has one, and its gradients are tested to match the existing execution path exactly. Note that a full `vi(backend="jax")` run currently still fails inside PyMC itself — for *any* model, not just HSSM models — due to two PyMC bugs (tracked in #1056). Until a PyMC release fixes those, run VI as before with the default backend or `backend="c"`.
3. **Fixed incorrect gradients for choice-only models with all-scalar parameters and outlier modeling.** For likelihoods that take parameters only (CPN/OPN networks) with no regression parameters, the gradient computation ignored any transformation applied downstream of the likelihood — for example the `p_outlier` mixture. Gradient-based fitting (VI, or NUTS on the default backend) of such models silently used incorrect gradients. The gradient is now exact, and a regression test asserts the chain rule holds on every backend.
4. **Internal cleanup of the LAN likelihood operators.** The Op classes are now defined once at module level instead of once per model, so building many models in one long-running process (e.g. a parameter-recovery loop in a notebook) no longer grows PyTensor's dispatch registry — which previously kept every model's network weights in memory for the lifetime of the process. The gradient hook was also migrated from PyTensor's deprecated `grad` to the new `pullback` API, silencing a `FutureWarning` that appeared on every model fit.

### 0.4.0

This version contains major breaking updates for HSSM. Please read the release notes below to migrate to HSSM 0.4.0.

#### Major new features:

1. A new `RLSSM` class has been added to support reinforcement learning sequential sampling models.

#### Breaking changes that require migration:

1. Dependencies have been streamlined to support PyMC 6.0+, pytensor 3.0+, ArviZ 1.0+, and Bambi 0.18+.
2. We added support for Python 3.14. However, `sample_posterior_predictive` sometimes fails due to a `cloudpickle` issue. Use Python 3.14 with caution if you have to perform posterior predictive sampling.
3. Consistent with PyMC 6.0+ and ArviZ 1,0+ expectations, the `model.sample()` by default uses `numba` as the compute backend.
4. `model.sample()` now returns an `xarray.DataTree` object instead of the `arviz.InferenceData` object. Other functions that expect `arviz.InferenceData` objects have been updated to accept `xarray.DataTree` objects.
5. `model.summary()` and `model.plot_trace()` methods are now removed. Use `az.summary()` and `az.plot_trace_dist()` instead.
6. HSSM can now be installed directly from PyPI via `pip` or `uv`. Conda support is no longer provided.

#### Bug fixes:

1. **Restore JAX-NUTS jitter control**: HSSM again disables the built-in initial-value jitter of the `numpyro`/`blackjax` samplers (PyMC 6 removed the public switch that made this possible), so sampling starts from HSSM's own controlled `initval_jitter` instead of an extra uniform jitter (#999).

### 0.3.1

This version includes the following changes:

1. **sbi NRE ONNX integration**: train an sbi Neural Ratio Estimator (NRE), export it to ONNX via `lanfactory.onnx.transform_sbi_to_onnx`, and load it into HSSM's `loglik_kind="approx_differentiable"` pipeline for MCMC inference (keystone tutorial `sbi_nre_integration.ipynb`).
2. **ONNX loader hardening**: `onnx2jax` now rejects ONNX graphs with dynamic/symbolic input dimensions (enforcing the single-trial + `jax.vmap` contract), and raises a clear error when JAX x64 is disabled and a graph carries int64 constants that would be truncated (guarding against silent likelihood corruption from flow-export sentinels). Applies across all ONNX exporters (LANfactory, BayesFlow, sbi).

### 0.3.0

This version includes the following changes:

1. Support for **choice-only models**: the HSSM class, data validator, distributions, and model configs now handle models without reaction times, including a softmax likelihood family.
2. **Racing Diffusion Model (RDM3)**: analytical likelihood, model configuration, and tests for a 3-choice RDM, with safe negative-RT handling and JAX backend support.
3. **Poisson Race model**: initial implementation of the Poisson race model.
4. **Hidden Markov Model (HMM-SSM)** example notebook.
5. **BayesFlow LRE integration** through HSSM for likelihood ratio estimation.
6. **RLSSM config system**: new `RLSSMConfig` and `BaseModelConfig` classes with comprehensive validation, plus a generalized RL likelihood builder supporting multiple computed parameters.
7. New tutorials: choice-only modeling, first HMM-SSM example, and updated existing tutorials.
8. `DataValidator` refactored to `DataValidatorMixin` for improved extensibility.
9. Bug fixes: default prior assignment, dimensionality errors with Bambi 0.17.0, negative-RT checks on missing data, flaky tests.
10. Infrastructure: Python 3.13 support, restructured CI test workflows with coverage reporting, updated `model.sample()` API to match Bambi conventions.

### 0.2.12

This version includes the following changes:

1. Compatibility with Python 3.13 and Bambi 0.17.1+.
2. A new `make_distribution_for_supported_model` convenience function for creating `pm.Distribution`.
3. Bug fix for default prior assignment.
4. Initial private implementations of RLSSM features.
5. Added disaggregation of quantile computation by grouping of choice.

### 0.2.11

This version include the following:

1. Simplification of simulator logic inside HSSM random variables (see the `ssm-simulators` `0.11.3` release as well)
2. Plotting functions now allow `prior_predictive` plots wherever suitable.
3. A new tutorial on using [bayeux](https://github.com/jax-ml/bayeux) for sampling
4. Consolidated `plotting` tutorial
5. New tutorial on how to use the `do-operator` from PyMC to control forward simulations

### 0.2.8

This version of HSSM incorporated the following changes:

1. Addition of tutorials for RLSSM paradigms.
2. Restructure `rldm.py` to make it more template-based to allow easy introduction of new models.

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

- Added an `hssm.plotting` submodule with `plot_predictive()` and
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
