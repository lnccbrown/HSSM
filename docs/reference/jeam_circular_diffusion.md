# JEAM circular diffusion

!!! warning "Experimental integration"

    `circular_diffusion` is a source-checkout prototype, not part of HSSM's stable
    released model surface. It depends on an immutable revision of the HSSM JEAM fork
    and has the limitations recorded below.

HSSM registers JEAM's two-dimensional, fixed-boundary circular diffusion model under
the model name `circular_diffusion`. It uses the ordinary [`hssm.HSSM`][hssm.HSSM]
class: the response domain, likelihood, and simulator are supplied by the model
configuration rather than a circular-model subclass.

## Install

From an HSSM source checkout that contains the integration, run:

```bash
uv sync --group jeam-prototype
```

The group pins the HSSM fork of JEAM to
[`0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce`](https://github.com/AlexanderFengler/JEAM/commit/0c0ef8b834dd062ad8aea5ff8e7a09dfb55492ce).
It is not installed with ordinary HSSM, and no public `hssm[jeam]` extra exists yet.
See [Installation](../getting_started/installation.md#experimental-circular-diffusion-with-jeam)
for the dependency-policy rationale.

## Model contract

The observed data must contain these columns in this order:

| Column | Meaning | Allowed values |
| --- | --- | --- |
| `rt` | Response time | Finite and strictly positive |
| `response` | Angular coordinate in radians | Finite values in the half-open interval $[-\pi, \pi)$ |

`response` is circular continuous data, not a categorical choice label. The pointwise
likelihood is a density with respect to the ordinary response-time and angular-coordinate
differentials, $d\,rt\,d\theta$.

The four HSSM parameters map to JEAM as follows:

| HSSM parameter | JEAM quantity | Configured bounds |
| --- | --- | --- |
| `v_x` | First Cartesian component of `drift_vec` | $(-3, 3)$ |
| `v_y` | Second Cartesian component of `drift_vec` | $(-3, 3)$ |
| `a` | Fixed decision threshold | $(0.1, 3)$ |
| `t` | Nondecision time (`ndt`) | $(0, 2)$ |

Parameters may be intercept-only or trial-wise through ordinary HSSM formulas. The
prototype fixes all other JEAM settings:

| Setting | Fixed value |
| --- | --- |
| Threshold dynamics | Fixed |
| Diffusion scale $\sigma$ | $1$ |
| Drift variability $s_v$ | $0$ |
| Nondecision-time variability $s_t$ | $0$ |
| Threshold decay | $0$ |
| Custom threshold function | None |

Construct the model with the lapse mixture disabled:

```python
import hssm

model = hssm.HSSM(
    data=data,
    model="circular_diffusion",
    p_outlier=None,
)
```

## Inference and prediction

The NumPy/Python `blackbox` likelihood remains the independent compatibility baseline.
It does not expose gradients to PyTensor, so it requires an explicitly gradient-free
step method such as one PyMC `pm.Slice` step for each parameter.

An opt-in, strict JAX likelihood is also registered through HSSM's ordinary
configuration surface:

```python
model = hssm.HSSM(
    data=data,
    model="circular_diffusion",
    loglik_kind="analytical",
    p_outlier=None,
)
```

This semi-analytical path evaluates JEAM's truncated first-passage series directly; it is
not a learned likelihood approximation. It preserves JEAM's pointwise values and exposes
reverse-mode parameter gradients through HSSM's existing JAX/PyTensor bridge.

For substantive fixed-CDM inference, the evidence-backed first recommendation is the
analytical likelihood with PyMC NUTS:

```python
traces = model.sample(
    sampler="pymc",
    target_accept=0.9,
)
```

NumPyro NUTS is also scientifically validated and was faster in the preregistered local
comparison. Variational inference has not been validated.

### Sampler capability matrix

“CI-supported” means that HSSM runs a real one-chain compilation smoke test and verifies
finite gradients, posterior variables, and sampler-specific statistics. It is weaker than
a recovery or efficiency recommendation.

| Likelihood | Sampler | Works | CI-supported | Recommended |
|---|---|---:|---:|---:|
| `blackbox` | PyMC Slice | yes | yes | yes, reference fallback |
| `analytical` | PyMC NUTS | yes | yes | yes, ordinary first recommendation |
| `analytical` | NumPyro NUTS | yes | yes | yes, validated alternative |
| `analytical` | BlackJAX | not enabled | no | no |
| `analytical` | nutpie | not enabled | no | no |
| `analytical` | Laplace | not enabled | no | no |

The likelihood choice does not silently change HSSM's global sampler policy. Select
`loglik_kind="analytical"` and either `sampler="pymc"` or `sampler="numpyro"`
explicitly. Unverified combinations fail before backend dispatch with the two currently
enabled choices in the error message.

Prior and posterior predictive sampling use JEAM's simulator through HSSM and accept
HSSM's seeded random-state interface. Draws preserve the final `[rt, response]` order.
The [complete marimo comparison lab](../tutorials/jeam_circular_diffusion.py) checks
both likelihoods against direct JEAM, compiles the analytical parameter gradient, and
lets users fit the same simulated dataset with blackbox/PyMC Slice, analytical/PyMC
NUTS, or analytical/NumPyro NUTS before running diagnostics and predictive checks.
The artifact-backed [NUTS recovery and efficiency report](../tutorials/jeam_nuts_recovery.py)
then shows the preregistered four-scenario recovery, diagnostics, predictive checks,
objective parity, timing decomposition, and 1,500-trial scaling evidence.

## Current boundary

The prototype supports only the fixed circular diffusion model described above. It does
not currently provide:

- continuous-response lapse or outlier mixtures (`p_outlier` must be `None`);
- native circular versions of HSSM's category-oriented plotting functions;
- automatic likelihood/sampler selection, a LAN likelihood, or a validated variational
  workflow;
- collapsing or custom thresholds, or nonzero $s_v$ or $s_t$;
- spherical, hyperspherical, projected, categorical, or ordinary continuous-response
  JEAM model registrations; or
- higher-dimensional observed responses beyond the two columns `[rt, response]`.

The wrapper has pointwise direct-JEAM/HSSM likelihood parity tests, deterministic
predictive tests, a marked-slow Bayesian recovery test, and a predeclared four-scenario
[repeated-recovery study](../tutorials/jeam_repeated_recovery.py). This evidence validates
the narrow handshake. A separate preregistered comparison supports PyMC and NumPyro NUTS
for the fixed CDM and promotes both over the Slice baseline on the measured local
efficiency gates. Neither study is a claim of long-run frequentist calibration or broad
hierarchical-workload performance.

## Promotion checklist

Before promoting the fixed-CDM integration to HSSM's stable released surface:

- [ ] JEAM publishes a versioned release containing the audited likelihood and seeded
  simulator contracts.
- [ ] HSSM replaces the source-only dependency group with a public, released optional
  extra such as `hssm[jeam]` and tests installation from built wheel metadata.
- [ ] Gradient-free inference is benchmarked on realistic hierarchical workloads, with
  documented runtime, convergence, and effective-sample-size expectations; otherwise a
  production-suitable likelihood strategy is selected.
- [ ] A density and simulator contract is designed and tested for lapse/outlier mixtures
  over continuous and circular responses.
- [ ] Native circular diagnostics and plotting cover prior/posterior prediction without
  treating angles as categorical choices.

Broader JEAM model families are a separate expansion gate. Each family must independently
specify its response coordinates and density measure, observed shape, parameter map,
simulation contract, numerical oracle tests, recovery evidence, and inference limitations
before it is registered in HSSM.
