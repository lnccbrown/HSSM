# Predictor centering and the intercept

A regression parameter such as `v ~ 1 + x` can show two intercept variables in
the trace: `v_Intercept_centered` and `v_Intercept`. This page explains where
they come from, how they relate, and which one to report. It also separates
predictor centering from a different mechanism with a similar name: the
centered or non-centered parameterization of group-specific terms.

## Two intercepts, one model

When a parameter's formula has an intercept and at least one common
(population-level) predictor, Bambi centers each predictor by subtracting its
mean before it builds the model. This is the `center_predictors=True` default.
Centering decorrelates the intercept from the slopes and usually makes sampling
easier. The model then contains:

- `v_Intercept_centered`, a sampled random variable: the value of `v`, on the
  link scale, when every common predictor is at its mean;
- `v_Intercept`, a deterministic on the original predictor scale: the value of
  `v` when every common predictor is zero.

They are related by

```text
v_Intercept = v_Intercept_centered - sum_k(mean(x_k) * v_x_k)
```

The slopes, such as `v_x`, are the same under both scales.

| Formula | Sampled intercept | Deterministic intercept |
|---|---|---|
| `v ~ 1` | `v_Intercept` | none |
| `v ~ 1 + (1 \| g)` | `v_Intercept` | none |
| `v ~ 1 + x` | `v_Intercept_centered` | `v_Intercept` |
| `v ~ 1 + x`, `center_predictors=False` | `v_Intercept` | none |
| `v ~ 0 + x` | none | none |

Intercept-only formulas, including those that add only group-specific terms,
have nothing to center and get no `_centered` variable. Categorical common
predictors are centered too, so `v ~ 1 + c` also produces both variables.

## The intercept prior keeps its meaning

A prior you set on `Intercept` applies to `v_Intercept_centered`, the intercept
at the predictor means. Bambi did this before version 0.20 as well. It sampled
the centered intercept, stored it under the name `v_Intercept`, and converted it
back to the original scale after sampling. Bambi 0.20 makes the same
distinction visible in the model graph and in the trace.

## Predictor centering is not non-centering

HSSM's `noncentered` argument controls something else. It selects how
group-specific (hierarchical) terms are parameterized:

- `noncentered=True` (the default) samples a standardized `v_1|g_offset` and
  records `v_1|g` as a deterministic, `v_1|g_offset * v_1|g_sigma`;
- `noncentered=False` samples `v_1|g` directly.

This setting affects group-specific terms only. It adds `*_offset` variables
and never a `_noncentered` suffix. Predictor centering affects only the common
intercept. The two settings are independent: `v ~ 1 + x + (1|g)` with the
defaults has both `v_Intercept_centered` and `v_1|g_offset`. See
[Centered vs. non-centered parameterizations](../tutorials/centered_vs_noncentered_basic_logic.ipynb)
for that choice.

## Which variable to report

Report `v_Intercept`, the intercept on the predictors' original scale, unless
you deliberately want the value at the predictor means. Both variables stay in
the trace, because Bambi's `predict` and `compute_log_likelihood` need the
centered one. To leave the centered variables out of a summary or a trace plot,
exclude them by name:

```python
import arviz as az

az.summary(idata, var_names=["~_centered"], filter_vars="like")
az.plot_trace(idata, var_names=["~_centered"], filter_vars="like")
```

To drop centering altogether, pass `center_predictors=False` to `hssm.HSSM`;
HSSM forwards it to Bambi. The intercept is then sampled on the original scale
under the name `v_Intercept`. This changes what the intercept prior refers to
and can make sampling harder when predictor means are far from zero.

## Initial values

PyMC ignores initial values for deterministics. If you pass
`model.sample(initvals={"v_Intercept": 0.5})` while `v` has common predictors,
HSSM moves the value to `v_Intercept_centered`, as it does for its own default
initial values. The two intercepts are equal when the slopes start at zero,
which is HSSM's default. A `v_Intercept_centered` key you pass explicitly takes
precedence. See [Set initial values](../tutorials/initial_values.ipynb).
