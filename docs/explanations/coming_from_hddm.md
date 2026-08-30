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

The parameterization also differs. HDDM's usual hierarchy is a group-only,
location-bearing distribution: with `v ~ 0 + (1 | participant_id)`, its free
group mean owns the population drift. This differs from
`v ~ 1 + (1 | participant_id)`, where the common intercept owns the population
location and the participant term is a zero-mean deviation.

HSSM requests the *non-centered* form by default, which often samples better.
Matching zero-mean deviations honor that request. With
`prior_settings="safe"`, however, a unique generated group-only term is centered
automatically so its free location remains connected to the likelihood; current
Bambi non-centering would otherwise discard it. HSSM reports this term-level
fallback, so setting the entire model `noncentered=False` is no longer necessary
merely to retain an HDDM-style generated group location. [Link functions and safe
priors](../tutorials/link_functions.ipynb) works through current
population-location ownership, link scale, and compatibility behavior from first
principles. [Specify hierarchical group
priors](../how_to/specify_group_priors.md) gives the explicit-prior rules and
parameterization overrides.

HSSM never changes an explicit group prior's meaning. If current Bambi cannot
construct it faithfully, HSSM instead raises before model construction. A group
prior must be hierarchical; numeric regression-term priors do not fix
coefficients. Under effective non-centering, Bambi currently supports only a
plain built-in Normal with hierarchical `sigma`, no extra arguments, and absent
or fixed-zero `mu`. To retain a free location or another prior family, make that
prior—and any hierarchical nested nodes—effectively centered. If the same
unmatched expression occurs under several grouping factors, add the exact common
term and use zero-mean deviations, or explicitly choose one location owner;
several free centered owners produce a likelihood ridge.

## Concepts with no direct translation

- **Priors.** For `ddm`, `ddm_sdv`, and `full_ddm`, HSSM's default priors on
  regression terms are derived from HDDM's — the settings are carried in
  `hssm.prior` under names like `HDDM_MU` and `HDDM_SIGMA` — so a regression
  model on these families starts from familiar ground. For safe common-intercept
  priors, an omitted link, `"identity"`, `bambi.Link("identity")`, and
  `hssm.Link("identity")` are equivalent: each keeps the intercept on the
  response scale and uses the HDDM-derived prior. A unique generated unmatched
  group intercept under identity likewise keeps its response-scale HDDM
  hierarchy and is centered so its location is retained. A transformed common
  intercept instead uses a coefficient-scale `Normal(mu=0, sigma=0.25)`, while
  a transformed unmatched group intercept uses a hierarchical Normal on the
  linear-predictor scale before the inverse link. For a nonlinear link,
  `g^-1(mu)` is a reference value, not generally the parameter's expectation.
  [Link functions and safe priors](../tutorials/link_functions.ipynb) explains
  these scale and ownership changes from first principles. Explicit priors are
  never rewritten, but incompatible group specifications are rejected before
  Bambi can drop or misinterpret them. For a generic identity-linked parameter,
  a unique generated group-only intercept with a finite configured bound now
  receives a centered native `TruncatedNormal` hierarchy. A pure group-intercept
  predictor is therefore supported by construction. The bound still applies to
  that coefficient, not to a complete additive predictor after slopes or other
  effects are added; HSSM warns about that distinction. A transformed link can
  instead constrain the assembled predictor when its inverse image matches the
  parameter support. Values outside configured likelihood bounds receive HSSM's
  finite per-trial log-likelihood floor (`-66.1`), not a hard-support rejection.
  [Specify hierarchical group priors](../how_to/specify_group_priors.md) gives
  the exact generated and explicit-prior rules. Analytical and black-box DDM
  variants keep their calibrated HDDM Gamma, Beta, or Normal hierarchies, while
  neural (`approx_differentiable`) variants use generic safe priors derived from
  the network's finite training bounds. Specifying your own is a different
  interface through the same prior controls.
- **Outliers.** HDDM's `p_outlier` exists in HSSM under the same name, and the
  lapse distribution is configurable rather than fixed. See [Model outliers
  with lapse probabilities](../tutorials/lapse_prob_and_dist.ipynb).
- **Sampling.** HSSM samples with PyMC and can dispatch to NumPyro and other
  JAX-based samplers. `model.sample()` passes keyword arguments through to
  PyMC, so `chains`, `draws`, `tune`, and `target_accept` behave as they do in
  PyMC rather than as in HDDM's sampler.
- **Results.** Posterior samples come back as an `xarray.DataTree` with
  ArviZ-compatible groups rather than an HDDM-specific container, so diagnostics,
  plots, and model comparison use the standard ArviZ/xarray ecosystem from there
  on.

## Where to start

If you are rebuilding an existing analysis, [the HSSM
tutorial](../tutorials/main_tutorial.ipynb) is the fastest route to the parts
you already know, in the new interface. The [Winterbrain 2025 workshop
snapshot](../archive/hssm_tutorial_workshop_2.ipynb) works through an analysis
first run in HDDM and then rebuilt in HSSM, including the data preparation
above.
