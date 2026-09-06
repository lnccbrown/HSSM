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
| `full_ddm` | `blackbox` | `blackbox` | `v`, `a`, `z`, `t`, `sz`, `sv`, `st` | `-1`, `1` |
| `angle` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `theta` | `-1`, `1` |
| `levy` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `alpha`, `t` | `-1`, `1` |
| `ornstein` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `g`, `t` | `-1`, `1` |
| `weibull` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `alpha`, `beta` | `-1`, `1` |
| `race_no_bias_angle_4` | `approx_differentiable` | `approx_differentiable` | `v0`, `v1`, `v2`, `v3`, `a`, `z`, `t`, `theta` | `0`, `1`, `2`, `3` |
| `ddm_seq2_no_bias` | `approx_differentiable` | `approx_differentiable` | `vh`, `vl1`, `vl2`, `a`, `t` | `0`, `1`, `2`, `3` |
| `gamma_drift` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `shape`, `scale`, `c` | `-1`, `1` |
| `gamma_drift_angle` | `approx_differentiable` | `approx_differentiable` | `v`, `a`, `z`, `t`, `theta`, `shape`, `scale`, `c` | `-1`, `1` |
| `lba3` | `analytical` | `analytical` | `A`, `b`, `v0`, `v1`, `v2` | `0`, `1`, `2` |
| `lba4` | `analytical` | `analytical` | `A`, `b`, `v0`, `v1`, `v2`, `v3` | `0`, `1`, `2`, `3` |
| `lba2` | `analytical` | `analytical` | `A`, `b`, `v0`, `v1` | `0`, `1` |
| `racing_diffusion_3` | `analytical` | `analytical` | `A`, `b`, `v0`, `v1`, `v2`, `t` | `0`, `1`, `2` |
| `poisson_race` | `analytical` | `analytical` | `r1`, `r2`, `k1`, `k2`, `t` | `-1`, `1` |
| `softmax_inv_temperature_2` | `analytical` | `analytical` | `beta`, `logit1` | `-1`, `1` |
| `softmax_inv_temperature_3` | `analytical` | `analytical` | `beta`, `logit1`, `logit2` | `0`, `1`, `2` |

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
