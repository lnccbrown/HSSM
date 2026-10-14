# Coming from HDDM

HDDM is HSSM's predecessor, and the workflow will feel familiar: you hand a
data frame of reaction times and responses to a model object, sample it, and
inspect the posterior. This page covers what changes, so that an analysis you
already understand can be rebuilt rather than re-derived.

There are three kinds of differences: the data your model expects, the way a
model is specified, and one parameter convention that will silently change
your numbers if you miss it.

## The convention that matters: `a`

**HSSM's `a` is the distance from the starting point to one boundary; HDDM's
`a` is the distance between the two boundaries.** An HDDM fit reporting
`a = 2.0` corresponds to `a = 1.0` in HSSM.

This is not a cosmetic difference. If you carry HDDM parameter values across —
as priors, as simulation ground truth, or when comparing published estimates —
divide them by two.

Halving also matters for a second reason: HSSM's bounds on `a` depend on which
likelihood you use. The analytical DDM leaves `a` on `(0, inf)`, but the
neural approximation is only valid over the range it was trained on, which for
`ddm` is `(0.3, 2.5)`. An HDDM `a` of `2.0` becomes a comfortable `1.0`; an
HDDM `a` of `6.0` becomes `3.0`, which is outside the trained range and calls
for the analytical likelihood instead.

You can see the conversion in HSSM's own source: the black-box likelihoods
wrap HDDM's Cython WFPT implementation and pass `a * 2` when they call it. The
other parameters — `v`, `z`, `t`, `sv`, `sz`, `st` — carry over unchanged, and
`z` is a relative starting point in `(0, 1)` in both packages.

## Your data

| HDDM | HSSM | Note |
|---|---|---|
| `subj_idx` | `participant_id` | Rename the column. It is the default grouping variable in formulas. |
| `response` coded `0` / `1` | `response` coded `-1` / `1` | Recode. HSSM's two-choice models use `-1` and `1`. |
| `rt` | `rt` | Unchanged, in seconds. |

In practice this is two lines before you build the model:

```python
data = data.rename(columns={"subj_idx": "participant_id"})
data["response"] = np.where(data["response"] == 0, -1, 1)
```

Unlike HDDM, HSSM does not require a participant column at all — a model with
no group-specific terms is fit to the pooled data. The column matters when you
write a formula that groups by it.

## Specifying a model

HDDM selects a model by class and configures it with keyword arguments
(`HDDMRegressor`, `depends_on`, `include`). HSSM has one class, and the model
family is a string:

```python
model = hssm.HSSM(data=data, model="ddm")
```

`model=` accepts `ddm`, `ddm_sdv`, `full_ddm`, `angle`, `levy`, `ornstein`,
`weibull`, and others; `hssm.list_models()` prints the current set. Models
without an analytical likelihood are served by a neural approximation that
HSSM downloads on first use, which is why the model family is a string rather
than a separate class.

The larger change is how effects on parameters are expressed. HDDM's
`depends_on` splits a parameter by condition; HSSM uses `lmer`-style formulas,
one per parameter, which covers the same ground and more:

| What you want | HDDM | HSSM |
|---|---|---|
| Drift varies by condition | `depends_on={'v': 'stim'}` | `include=[{"name": "v", "formula": "v ~ 0 + C(stim)"}]` |
| Drift varies by a continuous covariate | `HDDMRegressor` with a patsy formula | `"formula": "v ~ 1 + x"` |
| Per-participant drift | hierarchical by default | `"formula": "v ~ 1 + (1\|participant_id)"` |
| Several parameters, same structure | one specification per parameter | `global_formula="y ~ 1 + (1\|participant_id)"` |

The [hierarchical modeling tutorial](../getting_started/hierarchical_modeling.ipynb)
covers the formula syntax; [hierarchical DDM
regressions](../tutorials/ddm_hierarchical_tutorial.ipynb) works through
within-subject, between-subject, and interaction designs.

## Hierarchy is opt-in, and parameterized differently

HDDM models are hierarchical by default. In HSSM, a model is hierarchical
exactly when a parameter has a group-specific term in its formula — there is
no `hierarchical` switch to set.

The parameterization also differs. HDDM's convention is what HSSM calls the
*centered* form: drop the common intercept (`v ~ 0 + ...`) and let the
participant-level coefficients come from one group distribution with a free
mean and standard deviation. HSSM defaults to the *non-centered* form, which
usually samples better. To reproduce the HDDM specification, set
`noncentered=False`; the group distributions then appear as
`v_1|participant_id_mu`-style nodes. [Centered vs. non-centered
parameterizations](../tutorials/centered_vs_noncentered_basic_logic.ipynb)
explains the tradeoff and when each is the better choice.

## Concepts with no direct translation

- **Priors.** For `ddm`, `ddm_sdv`, and `full_ddm`, HSSM's default priors on
  regression terms are derived from HDDM's — the settings are carried in
  `hssm.prior` under names like `HDDM_MU` and `HDDM_SIGMA` — so a regression
  model on these families starts from familiar ground. The rule is the
  likelihood: those defaults apply unless you are using the neural
  (`approx_differentiable`) likelihood, which has its own priors derived from
  the network's training bounds. Specifying your own is a different interface
  — see [Specify priors and fix parameters](../how_to/specify_priors.ipynb).
- **Outliers.** HDDM's `p_outlier` exists in HSSM under the same name, and the
  lapse distribution is configurable rather than fixed. See [Model outliers
  with lapse probabilities](../tutorials/lapse_prob_and_dist.ipynb).
- **Sampling.** HSSM samples with PyMC and can dispatch to NumPyro and other
  JAX-based samplers. `model.sample()` passes keyword arguments through to
  PyMC, so `chains`, `draws`, `tune`, and `target_accept` behave as they do in
  PyMC rather than as in HDDM's sampler.
- **Results.** Posterior samples come back as an ArviZ `InferenceData` object
  rather than an HDDM-specific container, so diagnostics, plots, and model
  comparison are standard ArviZ from there on.

## Where to start

If you are rebuilding an existing analysis, [the HSSM
tutorial](../tutorials/main_tutorial.ipynb) is the fastest route to the parts
you already know, in the new interface. The [Winterbrain 2025 workshop
snapshot](../archive/hssm_tutorial_workshop_2.ipynb) works through an analysis
first run in HDDM and then rebuilt in HSSM, including the data preparation
above.
