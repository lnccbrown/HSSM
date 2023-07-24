# Changelog

## 0.1.x

### 0.1.2

- Improved numerical stability of `analytical` likelihoods.
- Added the ability to handle lapse distributions.
- Added the ability to perform prior predictive sampling
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
