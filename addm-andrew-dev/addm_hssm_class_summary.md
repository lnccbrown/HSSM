# aDDM-in-HSSM (subclass plan) — at a glance

Plain-language companion to [addm_hssm_class.md](addm_hssm_class.md). That file is the dense,
implementation-ready version; this one is the map. Read this to understand *what each commit does
and why*; read the full plan when you're writing the code.

## The one-paragraph version

We add the **attentional drift diffusion model (aDDM)** to HSSM as **`hssm.aDDM(...)`**, a
subclass of `HSSMBase` built the same way the existing `RLSSM` is. We copy ("vendor") the fast
JAX likelihood from the sibling `efficient-fpt` repo into HSSM, wrap it in a differentiable
PyTensor `Op` so NUTS gets gradients for free, and give aDDM its own config object and data
checks. Then we add **non-decision time** as a sampled parameter, and teach the **simulator** to
use a subject's real fixations and to *keep saccading* past the last observed fixation so
posterior predictive checks aren't biased. Each commit ships with its own tests.

## Why a subclass (not `model="addm"`)

We follow `RLSSM` so aDDM sits next to it as a peer —
entry point `hssm.aDDM(data=...)`, **not** `hssm.HSSM(model="addm", ...)`. The payoff: **we touch
almost no generic HSSM code** (no edits to the model registry, the global validator, or the base
classes). aDDM is simpler than RLSSM in one way — trials are independent, so there's no
participant-by-trial reshaping and no balanced-panel requirement.

## What you get to write at the end

```python
import hssm
model = hssm.aDDM(
    data=addm_trials,                     # rt, response + r1, r2, flag, sacc_array, d
    model_config=hssm.aDDMConfig(),       # or just omit for defaults
    include=[{"name": "eta", "formula": "eta ~ 1 + (1|participant)"}],
)
idata = model.sample()
```

Sampled parameters: `eta, kappa, a, b, x0` (+ `t` after Commit 5).
Per-trial data (not sampled): `r1, r2, flag, sacc_array, d, sigma`.

## The commits

| # | Commit | One-liner | How we know it works |
|---|--------|-----------|----------------------|
| 1 | Vendor the JAX likelihood | Copy efficient-fpt's `jax/` kernel into `hssm/addm/likelihoods/jax/`; fix the cross-package imports. | Vendored kernel runs on a tiny batch and matches the per-trial reference to `1e-6`. |
| 2 | Builder + Op + attention process | Wrap the kernel as a `logp` and a gradient-carrying PyTensor `Op`; add a pluggable "attention process" (default = the standard alternating drift). | Op = func = raw kernel to `1e-6`; gradients finite; a custom attention process changes the answer as expected. |
| 3 | `aDDMConfig` dataclass | A config object (params, bounds, extra fields, attention process) mirroring `RLSSMConfig`. | Defaults are right; `validate()` accepts good configs and rejects bad ones; dict round-trips. |
| 4 | `aDDM(HSSMBase)` subclass | The real class: builds the Op in `__init__`, validates aDDM columns, exports `hssm.aDDM`. | `hssm.aDDM(data=df)` constructs, is an `HSSMBase`, and a 5-draw smoke sample returns `InferenceData`. |
| 5 | Non-decision time `t` | Add `t` as a sampled pre-decision delay; shift **both** RT and fixation onsets into "decision time". | `t=0` reproduces Commit 4 exactly; `t>0` matches a manual shift; impossible `t` is rejected per-trial (returns `-inf`); gradient w.r.t. `t` is finite. |
| 6 | Simulator: real gaze + keep saccading (cross-repo) | Teach `cssm.addm` to use the dataset's covariates and to **continue generating fixations** past the last observed one, so PPC isn't frozen at the final drift. | Supplied covariates are used verbatim; sequences extend past the prefix and always cover the decision horizon; no covariates = unchanged old behavior. |
| 7 | Recovery + tutorial + docs | Off-CI parameter-recovery script; a tutorial notebook; README/nav updates. | Posterior means land within ~2σ of ground truth on 1000 trials; the notebook executes; `mkdocs build` succeeds. |
| 8 | Cleanup | Delete the scratch folder; bump the ssm-simulators floor; refresh the vendor hash. | Scratch dir gone; installed simulator meets the new floor. |

## The two ideas worth understanding

**Non-decision time is not just "subtract `t` from RT" (Commit 5).** The fixation onsets
(`sacc_array`) are recorded from the start of the *trial*, but the likelihood reasons in
*decision time*. So when we remove the first `t` seconds of motor/encoding delay, we shift the RT
*and* slide every fixation onset back by `t` (keeping the first one anchored at 0). We require `t`
to fall inside the first fixation (true in practice: NDT ≈ 150–400 ms, first fixations ≈ 500 ms+),
which keeps the math smooth and the gradients clean. Any proposed `t` that breaks this just gets
rejected by the sampler instead of crashing it.

**The simulator's "last-fixation freeze" (Commit 6).** Today the simulator pre-generates a big
fixed budget of fixations and, if a slow decision runs out of them, freezes the drift at the last
fixation's value. That's masked when the simulator invents its own (over-long) fixations, but it
**breaks** the moment we feed it a real subject's fixations, which only run up to their observed
response. The fix: always treat the subject's fixations as a *prefix*, then keep drawing new
fixations by fitting a subject-wise Gamma distribution that appends additional fixations until the whole decision is
covered or a preset max decision time. Because the decision boundary collapses over time, "the whole decision" is a finite
horizon, so coverage is guaranteed and the freeze becomes impossible. This is what makes
posterior predictive checks trustworthy.


