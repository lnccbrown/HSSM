# How-to guides

Use these guides when you already know what you want to accomplish. Each route
focuses on a concrete modeling, inference, analysis, or extension task.

## Build and parameterize a model

- [Specify priors and fix parameters](specify_priors.ipynb).
- [Use stimulus coding](../tutorials/tutorial_stim_coding.ipynb).
- [Add smooth effects with `hsgp()`](../tutorials/hsgp_regression.ipynb).
- [Model outliers with lapse probabilities](../tutorials/lapse_prob_and_dist.ipynb)
  or [regress on `p_outlier`](../tutorials/tutorial_p_outlier_regression.ipynb).

## Choose and run an inference method

- [Set initial values](../tutorials/initial_values.ipynb).
- [Use an alternative sampler with Bayeux](../tutorials/tutorial_bayeux.ipynb).
- [Run variational inference](../tutorials/variational_inference.ipynb).

## Work with fitted models

- [Plot posteriors and predictions](../tutorials/plotting.ipynb), then consult
  the [posterior-predictive](../tutorials/ppc_gallery.ipynb) or
  [model-cartoon](../tutorials/cartoon_gallery.ipynb) gallery.
- [Compare and interpret models](compare_models.ipynb).
- [Extract trial-wise parameters](../tutorials/tutorial_trial_wise_parameters.ipynb).
- [Run a Bayesian t-test on posterior draws](../tutorials/tutorial_bayesian_t_test.ipynb).
- [Simulate interventions with the do-operator](../tutorials/do_operator.ipynb).
- [Save and load fitted models](../tutorials/save_load_tutorial.ipynb).

## Extend HSSM

Begin with the [custom-likelihood route table](external_trainers.md). For ONNX,
read the [ONNX likelihood contract](custom_onnx_likelihoods.ipynb) before using
the sbi, BayesFlow, JAX-callable, or black-box walkthrough linked there.

For a contribution to HSSM itself, follow the
[local development setup](../local_development.md) and then the
[contribution guide](../CONTRIBUTING.md).

For concepts rather than procedures, return to the
[learning index](../learn/index.md). For exact APIs, use the
[reference index](../reference/index.md).
