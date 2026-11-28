# Built-in models and likelihoods

`hssm.HSSM(model=...)` accepts the built-in model names below. The table is
tested against `hssm.list_models()` and each model's default configuration, so a
code change that leaves this catalog stale fails CI.

When `loglik_kind` is omitted, HSSM prefers `analytical`, then
`approx_differentiable`, then `blackbox` among the kinds configured for that
model.

| Model | Available likelihood kinds | Default | Parameters | Choices |
| --- | --- | --- | --- | --- |
| `ddm` | `analytical`, `approx_differentiable`, `blackbox` | `analytical` | `v`, `a`, `z`, `t` | `-1`, `1` |
| `ddm_sdv` | `analytical`, `approx_differentiable`, `blackbox` | `analytical` | `v`, `a`, `z`, `t`, `sv` | `-1`, `1` |
| `ddm_st` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `st` | `-1`, `1` |
| `full_ddm` | `blackbox` | `blackbox` | `v`, `a`, `z`, `t`, `sz`, `sv`, `st` | `-1`, `1` |
| `full_ddm_lan` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `sz`, `sv`, `st` | `-1`, `1` |
| `angle` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `theta` | `-1`, `1` |
| `levy` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `alpha`, `t` | `-1`, `1` |
| `ornstein` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `g`, `t` | `-1`, `1` |
| `weibull` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `alpha`, `beta` | `-1`, `1` |
| `race_no_bias_angle_4` | `approx_differentiable` | `approx_differentiable` | `v0`, `v1`, `v2`, `v3`, `a`, `z`, `t`, `theta` | `0`, `1`, `2`, `3` |
| `ddm_seq2_no_bias` | `approx_differentiable` | `approx_differentiable` | `vh`, `vl1`, `vl2`, `a`, `t` | `0`, `1`, `2`, `3` |
| `gamma_drift` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `k`, `scale`, `c` | `-1`, `1` |
| `gamma_drift_angle` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `theta`, `k`, `scale`, `c` | `-1`, `1` |
| `lba3` | `analytical` | `analytical` | `A`, `b`, `v0`, `v1`, `v2` | `0`, `1`, `2` |
| `lba4` | `analytical` | `analytical` | `A`, `b`, `v0`, `v1`, `v2`, `v3` | `0`, `1`, `2`, `3` |
| `lba2` | `analytical` | `analytical` | `A`, `b`, `v0`, `v1` | `0`, `1` |
| `racing_diffusion_3` | `analytical` | `analytical` | `v0`, `v1`, `v2`, `A`, `b`, `t` | `0`, `1`, `2` |
| `poisson_race` | `analytical` | `analytical` | `r1`, `r2`, `k1`, `k2`, `t` | `-1`, `1` |
| `circular_diffusion` | `analytical` | `analytical` | `v_x`, `v_y`, `a`, `t` | Angle in radians, [−π, π) |
| `softmax_inv_temperature_2` | `analytical` | `analytical` | `beta`, `logit1` | `-1`, `1` |
| `softmax_inv_temperature_3` | `analytical` | `analytical` | `beta`, `logit1`, `logit2` | `0`, `1`, `2` |
| `softmax_inv_temperature_4` | `analytical` | `analytical` | `beta`, `logit1`, `logit2`, `logit3` | `0`, `1`, `2`, `3` |

### Circular reports

The [fixed-boundary circular diffusion model](https://doi.org/10.1037/rev0000023)
starts at the center of a disk of radius `a` with isotropic unit diffusion
and Cartesian drift `(v_x, v_y)`. Boundary contact determines the reported
angle. Decision time plus nondecision time `t` gives the RT. Supply ordinary
`rt` and `response` columns; the latter contains a continuous angle, not a
choice code. The native analytical likelihood is differentiable and requires
neither a likelihood network nor an external research package.

```python
hssm.set_floatX("float64")
model = hssm.HSSM(data=data, model="circular_diffusion", p_outlier=None)
fit = model.sample(sampler="pymc", random_seed=42)
predictions = model.sample_posterior_predictive(dt=fit, inplace=False, random_seed=43)
```

Keep lapse mixtures off. Use `plot_predictive` for pooled RTs, and use the
raw predictive angles for directional summaries. Quantile-probability plots
and model cartoons require categorical choices. The simulator and likelihood use
a fixed radius, with no across-trial variability in drift or nondecision time
unless modeled explicitly through covariates. This is a planar model with a
single exit angle, not a spherical diffusion model.

## Specialized model families

The table covers the `hssm.HSSM(model=...)` registry. Two specialized public
classes have separate configuration and discovery surfaces:

- [`hssm.aDDM`](../api/addm.md) models attentional evidence accumulation from
  trial and fixation data.
- [`hssm.RLSSM`](../api/rl.md) combines reinforcement-learning updates with an
  SSM observation model; its class-level and module-level `list_models`
  functions enumerate the RL registry.

Use [`hssm.list_models`](../api/model_registry.md) to discover the built-in HSSM
names programmatically. Custom HSSM configurations can be added with
[`hssm.register_model`](../api/model_registry.md).

For the conceptual difference between the likelihood kinds, see
[Likelihood kinds in HSSM](../explanations/likelihoods.md).
