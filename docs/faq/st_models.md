# My `ddm_st` or `ddm_normal_st` fit won't converge, or converges to the wrong answer

Both models add trial-to-trial variability in non-decision time to the DDM, and both
are fitted through a likelihood approximation network (LAN), a neural network
trained to return the model's log-density. Their failures rarely raise an error.
Usually the chains stick, or worse, the fit converges cleanly and recovers the wrong
values. Work through the checks below in order; the first two are the most common.

## The two models

On each trial the non-decision time `tau` is drawn from a kernel centred on `t`, and
the response time is that plus the first-passage time of a Wiener process:

```text
ddm_st:         tau ~ Uniform(t - st, t + st)      st is a half-width
ddm_normal_st:  tau ~ Normal(t, st)                st is a standard deviation

(rt, response) = tau + first passage of a Wiener process with drift v,
                 boundary separation a, starting point z * a
```

The same number means a different amount of variability in the two models. A
uniform kernel with half-width `s` has standard deviation `s / sqrt(3)`, so
`st = 0.2` in `ddm_st` is about as much spread as `st = 0.12` in `ddm_normal_st`.
Do not compare `st` estimates across the two.

HSSM evaluates, for each trial,

```text
p(rt, response) = (1 - p_outlier) * f_LAN(rt, response | v, a, z, t, st)
                  + p_outlier * Uniform(rt; 0, 20)
```

where `f_LAN` is the network's density, set to a floor value for response times
below the model's support edge (next section).

## 1. Use an HSSM that knows where the support starts

Non-decision time can fall below `t` in both models, so response times can too.
HSSM floors the likelihood of any response faster than the model's support edge,
and that edge has to be in the right place:

| model | support starts at | why |
|---|---|---|
| `ddm_st` | `t - st` | the uniform kernel's lower edge |
| `ddm_normal_st` | `t - 3 * st` | the Normal kernel has no edge; three SDs below `t` is the practical one |

Both models declare their edge (`ndt_edge_shift` in the model config), and HSSM
0.5.1 and later honour it (#1292). Earlier versions floor every response faster
than `t`. For these models that discards real density. The fit runs
without an error, but `t` comes out biased, and the result looks like awkward
sampler geometry rather than a wrong model.

## 2. Stay inside the network's training box

A LAN is only accurate inside the parameter region it was trained on. Outside that
region it still returns finite, plausible numbers, and the sampler will use them.
Both networks were trained on:

| `v` | `a` | `z` | `t` | `st` |
|---|---|---|---|---|
| -3 to 3 | 0.3 to 2.5 | 0.3 to 0.7 | 0.25 to 2.25 | 0.001 to 0.35 |

The registered models use these bounds already. If you narrow them through
`model_config={"bounds": ...}`, stay inside this box; do not widen it.

If you fit simulated data, remove response times at or below zero before fitting.
HSSM refuses them, and both simulators produce them in the corner of the box where
`st` is large and `t` small: about 4% of trials for `ddm_st` and 16% for
`ddm_normal_st` at `t = 0.25`, `st = 0.35`, `a = 0.3`, `v = 0`.

```python
sim = ssms.basic_simulators.simulator.simulator(model="ddm_st", theta=theta, n_samples=n)
data = pd.DataFrame({"rt": sim["rts"].ravel(), "response": sim["choices"].ravel()})
data = data[data["rt"] > 0]
```

## 3. Hierarchical fits: widen the prior on the group means

This is the failure that looks like success. For a hierarchical model such as

```python
include = [{"name": p, "formula": f"{p} ~ 0 + (1|subject)"} for p in ["v", "a", "z", "t"]]
model = hssm.HSSM(data, model="ddm_st", include=include,
                  link_settings="log_logit", prior_settings="safe", noncentered=False)
```

each parameter is modelled on the link scale as

```text
logit-scaled p[subject] ~ Normal(mu_p, sigma_p)
mu_p    ~ Normal(0, 0.25)      HSSM's default for a group intercept under log_logit
sigma_p ~ Weibull(1.5, 0.3)
```

and `inv_link(0)` is the midpoint of the parameter's bounds. A `Normal(0, 0.25)`
prior on `mu_p` pulls each group mean toward that midpoint. With 12 subjects and 300
trials each, that pull dominates for `v` and `t`. Every convergence diagnostic stays
clean, and the per-subject estimates look fine, because each subject's own trials
pin them down. The error is only visible at the group level:

| 12 subjects x 300 trials | truth | `mu_p ~ Normal(0, 2)` | `mu_p ~ Normal(0, 0.25)` (default) |
|---|---|---|---|
| `v` population mean, `ddm_st` | 1.22 | 1.20 to 1.27 | **0.68 to 0.72** |
| `t` population mean, `ddm_st` | 0.39 | 0.39 | **0.65 to 0.66** |
| `t` population SD, `ddm_st` | 0.057 | 0.056 to 0.057 | **0.23 to 0.24** |
| `v` population mean, `ddm_normal_st` | 1.22 | 1.20 to 1.27 | **0.65 to 0.68** |
| `t` population mean, `ddm_normal_st` | 0.39 | 0.39 | **0.65 to 0.66** |
| `t` population SD, `ddm_normal_st` | 0.057 | 0.053 to 0.056 | **0.23 to 0.24** |

Each range spans the two lapse settings in section 4. Every fit in the table had
r-hat at most 1.006 and at most 0.03% divergent transitions. `a` and `z` survive the
default in this example only because their true group means sit near the midpoints
of their bounds. That does not make the default safe for them.

The fix is an explicit, wider prior on the group mean:

```python
import bambi as bmb

group_prior = bmb.Prior(
    "Normal",
    mu=bmb.Prior("Normal", mu=0.0, sigma=2.0),
    sigma=bmb.Prior("Weibull", alpha=1.5, beta=0.3),
)
include = [
    {"name": p, "formula": f"{p} ~ 0 + (1|subject)", "prior": {"1|subject": group_prior}}
    for p in ["v", "a", "z", "t"]
]
```

Keep `st` as a single parameter shared by all subjects rather than adding it to
`include`; a separate `st` per subject is poorly constrained by a few hundred
trials.

## 4. Choose `p_outlier` to match the data

`p_outlier` mixes a uniform "lapse" distribution into every trial, as in the
likelihood above. HSSM's default is 0.05.

- **Simulated data without lapses:** use `p_outlier=None` (or 0). Fitting a lapse
  term the data do not have is a fixed misspecification. Its bias does not shrink
  with more data, so coverage gets worse as datasets grow. LAN_pipeline_minimal's
  [recovery guide](https://github.com/lnccbrown/LAN_pipeline_minimal/blob/main/docs/how-to/run-recovery-sweep.md)
  measures this on the plain DDM with its exact likelihood over 12
  datasets: 94% intervals covered the truth 96% of the time without the lapse, and
  88% with 0.05, at 500 trials. In the recovery above the lapse-free fits also put
  the `v` group mean closer to the truth (1.20 against 1.27, truth 1.22), for both
  models.
- **Real data:** keep the lapse term. Real participants do lapse, and without the
  term a handful of very fast or very slow responses can drag the DDM parameters.

The lapse is not a convergence lever. If chains are stuck, changing `p_outlier` does
not fix the cause, so look at sections 1 to 3 first.

## 5. Clean diagnostics do not mean a correct answer

r-hat, effective sample size and divergences tell you whether the chains agree. They
do not tell you whether the answer is right, as section 3 shows. Look at the trace
plots: chains that agree but sit against a bound, or a posterior with two peaks, show
up there and not in the summary table. Where each network is known to be weaker
than the exact model is listed in its model card on the Hugging Face Hub. In short:

- `ddm_normal_st` reads `st` about 15% low when the true `st` is between 0.05 and
  0.11, and that offset does not shrink with more trials.
- With thousands of trials per subject, both networks return posteriors narrower
  than the exact likelihood would, a known property of LANs at that scale.
- `st` and `t` trade off against each other: a larger `st` can be absorbed by a
  larger `t`. On a single dataset you can see both come out high together.

## Which advice applies to which model

Everything above applies to both models, with three differences:

| | `ddm_st` | `ddm_normal_st` |
|---|---|---|
| `st` is | the half-width of a uniform kernel | the SD of a Normal kernel |
| support starts at | `t - st` | `t - 3 * st` |
| simulated `rt <= 0` at the box corner | about 4% | about 16% |
