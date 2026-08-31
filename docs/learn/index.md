# Learn HSSM

Use this section for guided material that builds understanding through complete
examples. If you are new to HSSM, follow the first path in order; the remaining
paths assume that you can already fit and inspect a basic model.

## Start with a complete workflow

1. [Install HSSM](../getting_started/installation.md).
2. [Fit and check a first DDM](../getting_started/getting_started.ipynb).
3. [Work through the main HSSM tutorial](../tutorials/main_tutorial.ipynb).
4. [Add hierarchical structure](../getting_started/hierarchical_modeling.ipynb).
5. [See one analysis from data to interpretation](../tutorials/scientific_workflow_hssm.ipynb).

## Learn a model family

- [Hierarchical DDM regressions](../tutorials/ddm_hierarchical_tutorial.ipynb)
  connect an experimental design to parameter formulas.
- [Choice-only models](../tutorials/choice_only_models.ipynb) cover decisions
  without response times.
- [Poisson race models](../tutorials/poisson_race.ipynb) introduce a
  multi-accumulator model.
- [HMMs with DDM emissions](../tutorials/hmm_ddm_regime_switching.ipynb) model
  regime changes across trials.
- [Attentional DDMs](../tutorials/attentional_ddm.ipynb) connect fixation
  covariates to evidence accumulation.

## Learn reinforcement-learning models

Start with [RLSSM basics](../tutorials/rlssm_basic.ipynb), then choose the route
that matches your data:

- [choice-only RLSSMs](../tutorials/choice_only_rlssm.ipynb);
- [custom RLSSMs with `ssms.rl`](../tutorials/rlssm_advanced.ipynb);
- [restless learners](../tutorials/rlssm_restless_learner.ipynb); or
- [registering a custom RLSSM](../tutorials/rlssm_hssm_custom_models.ipynb).

## Understand modeling choices

- [Coming from HDDM](../explanations/coming_from_hddm.md) maps familiar HDDM
  concepts to HSSM.
- [Likelihood kinds in HSSM](../explanations/likelihoods.md) explains the
  available likelihood kinds.
- [Centered and non-centered parameterizations](../tutorials/centered_vs_noncentered_basic_logic.ipynb)
  establish the basic trade-off.
- [Random-slope prior diagnostics](../tutorials/random_slope_safe_priors.ipynb)
  show how parameterization decisions affect real models.

Ready to solve a specific problem? Continue to the
[how-to guide index](../how_to/index.md). For exact signatures and supported
helpers, use the [reference index](../reference/index.md).
