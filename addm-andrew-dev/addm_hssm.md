# Plan: Integrating aDDM into HSSM

Branch: `addm-integration-0` (HSSM repo).

## Context

The attentional drift diffusion model (aDDM; Krajbich et al.) extends the standard DDM by modulating the drift rate based on **which option the subject is currently fixating**. Two upstream packages already give us most of what we need:

1. **efficient-fpt** (sibling repo) — provides a fast, differentiable **JAX batch log-likelihood** for the aDDM. Recently restructured: there is no longer a separate `efficient_fpt_jax` package; everything lives under a unified `efficient_fpt/` with `jax/`, `numpy/`, and `cython/` backend subpackages.
2. **ssm-simulators** — HSSM's standard simulator dependency. It now ships a Cython aDDM simulator: `addm` is already registered in [ssms/config/_modelconfig/addm.py](/users/azhan378/ssm-simulators/ssms/config/_modelconfig/addm.py) pointing to `cssm.addm`. HSSM therefore gets the **simulator path (prior/posterior predictives) for free** once we pin the ssm-simulators release that contains the registration.

Goal: aDDM becomes a **first-class registered model in HSSM**, on equal footing with `ddm`, `angle`, `weibull`, etc. Users write

```python
model = hssm.HSSM(model="addm", data=addm_trial_df, include=[...])
model.sample()
```

where `addm_trial_df` has standard SSM columns (`rt`, `response`) plus aDDM-specific per-trial covariates (`r1`, `r2`, `sacc_array`, `d`, `flag`). The aDDM needs per-trial covariates that are *not* sampled parameters — the same pattern RLSSM solves — so we reuse HSSM's existing extra-fields machinery.

### Design: aDDM as a registered model, **not** a class

> **Architectural pivot from prior drafts.** Earlier versions of this plan proposed an `aDDM(HSSMBase)` peer class with its own `aDDMConfig`. We are abandoning that and instead following the **`hssm.HSSM(model="<name>", ...)`** registration pattern used by every other built-in model.
> All aDDM-specific behavior is still encapsulated — but in functions and a config dict rather than a class hierarchy.

Why this works cleanly: HSSM's `LogLik` type ([`src/hssm/_types.py:11`](src/hssm/_types.py#L11)) is `Union[str, PathLike, Callable, Op, type[Distribution]]`, so the `loglik` field of a registered model config can be a **pre-built PyTensor `Op` wrapping the JAX likelihood**. We build that `Op` once at module import time (inside `addm_config.py`) and stamp it into the config dict. HSSM's existing extra-fields plumbing ([`data_validator.py:156`](src/hssm/data_validator.py#L156)) already threads the per-trial covariates into the `Op`'s `*args`. The result is that `HSSM(model="addm", ...)` "just works" — no override, no subclass, no special-case code path inside `HSSM.__init__`.

The only piece that does not fit cleanly into the registered-model mold is **data-shape validation for `sacc_array`** (it is a 2-D array stored inside a DataFrame column, which the generic data validator does not check today). That gets a small model-name-gated hook in `data_validator.py`.

### Use cases for efficient-fpt (what HSSM consumes, and what we do not)

efficient-fpt now exposes several entry points; the plan picks the right one for each use case:

| Use case | Function (post-restructure) | Notes |
|---|---|---|
| **HSSM likelihood for NUTS (primary)** | `efficient_fpt.jax.compute_addm_loglikelihoods` (alias to `compute_addm_loglikelihoods_batchscan`) | Production batched JAX kernel. One trace through `lax.scan` over the whole batch. This is what the registered `loglik` `Op` wraps. |
| Optimizer/NLL closure (reference only) | `efficient_fpt.jax.make_addm_nll_function` | Convenience wrapper for fitting scripts; **not** what HSSM uses — HSSM needs per-trial log-likes summed by PyMC, not a pre-built mean NLL. Useful as a parity check. |
| Single-trial JAX (parity tests only) | `efficient_fpt.jax.compute_addm_logfptd` (alias to `compute_addm_logfptd_precomputed`) | Used in unit tests to confirm batched output matches per-trial output. **Not** used in the inference path. |
| Stage-scan single-trial variant | `efficient_fpt.jax.compute_addm_logfptd_stagescan` | Algorithm-comparison kernel. Not used by HSSM. |
| Drift-array helper (JAX) | `efficient_fpt.jax.addm_helpers._build_addm_mu_array_data` | Already JAX-vmapped; converts `(eta, kappa, r1, r2, flag, d, max_d)` → padded `(n_trials, max_d)` drift array. HSSM uses this directly — **no need to reimplement an attention/learning process module**. |
| Precision control | `efficient_fpt.jax.set_jax_precision`, `get_jax_dtype` | Call once at import time; do **not** auto-mutate JAX precision on import. |
| **Simulator** | *Do not import from efficient-fpt* | HSSM's simulator surface goes through ssm-simulators (`cssm.addm`). efficient-fpt's Cython simulator (`_simulate_addm_fpt`) is heavyweight to build; ssm-simulators already covers this. |
| Non-JAX backends | `efficient_fpt.cython.*`, `efficient_fpt.numpy.*` | Used by efficient-fpt's `aDDModel` for non-autodiff fitting. Irrelevant to HSSM. |
| `efficient_fpt.aDDModel` | High-level user-facing class | Useful for testing/simulation **outside** HSSM. We import it only in tests as a dev dependency to generate synthetic data; HSSM itself does not depend on it. |

### Integration strategy: vendor the JAX likelihood, depend on ssm-simulators

- **Vendor** (copy into HSSM, no new runtime dependency): the JAX likelihood code from `efficient_fpt/jax/`. efficient-fpt is not on PyPI, its Cython compile chain is heavyweight, and HSSM already depends on `jax`/`jaxlib`. Vendoring lets HSSM ship a frozen, audited copy that evolves on HSSM's release cadence.
- **Depend** (no copy): ssm-simulators is already a transitive dependency. With `addm` registered there, HSSM's existing `hssm.simulate_data(...)` flow works for aDDM out of the box once the ssm-simulators release containing `addm` is pinned.

---

## Plan, organized by commits to `addm-integration-0`

Each commit below is self-contained, reviewable, and passes tests on its own. Branch off `main` at `f2650fd5` (current HEAD).

### Commit 1 — `vendor: copy efficient-fpt JAX likelihood into hssm.addm.likelihoods.jax`

**Source files to copy** (verbatim, from efficient-fpt `src/efficient_fpt/jax/`):

| Source | Destination |
|---|---|
| `efficient_fpt/jax/multi_stage.py` | `src/hssm/addm/likelihoods/jax/multi_stage.py` |
| `efficient_fpt/jax/single_stage.py` | `src/hssm/addm/likelihoods/jax/single_stage.py` |
| `efficient_fpt/jax/batch.py` | `src/hssm/addm/likelihoods/jax/batch.py` |
| `efficient_fpt/jax/addm_helpers.py` | `src/hssm/addm/likelihoods/jax/addm_helpers.py` |
| `efficient_fpt/jax/utils.py` | `src/hssm/addm/likelihoods/jax/utils.py` |
| `efficient_fpt/_defaults.py` (relevant constants only) | `src/hssm/addm/likelihoods/jax/_defaults.py` |
| `efficient_fpt/utils.py:resolve_quadrature_orders` | inline into `jax/utils.py` |

**Actions in this commit:**

1. Create `src/hssm/addm/likelihoods/jax/` and copy the six files. Relative imports inside the copied files (`from .._defaults`, `from ..utils import resolve_quadrature_orders`) must be retargeted because `_defaults.py` and `resolve_quadrature_orders` were one level up in efficient-fpt:
   - `from .._defaults` → `from ._defaults` (now sibling).
   - `from ..utils import resolve_quadrature_orders` → import from local `utils` (we inline this small helper).
2. Header comment in each copied file recording the upstream commit hash at copy time, so future maintainers can diff against upstream.
3. Create `src/hssm/addm/likelihoods/jax/__init__.py` exposing only what HSSM needs downstream:
   ```python
   from .batch import compute_addm_loglikelihoods, make_addm_nll_function
   from .multi_stage import compute_addm_logfptd  # parity tests only
   from .addm_helpers import _build_addm_mu_array_data
   from .utils import set_jax_precision, get_jax_dtype
   ```
4. `src/hssm/addm/likelihoods/jax/NOTICE` with efficient-fpt's license text and a `Source-Commit: <sha>` line.

**Not vendored:** `efficient_fpt/cython/`, `efficient_fpt/numpy/`, `efficient_fpt/models.py`, `efficient_fpt/addm_helpers.py` (the NumPy version), the simulator. None used by inference.

No code outside `src/hssm/addm/` should import from `hssm.addm.likelihoods.jax` directly; keeping the surface narrow makes a future re-vendor a single-file change.

### Commit 2 — `feat(addm): JAX logp builder and PyTensor Op wrapper`

Create `src/hssm/addm/__init__.py` (empty marker — nothing publicly exported here, since aDDM has no user-facing class) and `src/hssm/addm/likelihoods/__init__.py`.

Create **`src/hssm/addm/likelihoods/builder.py`**, mirroring [hssm/rl/likelihoods/builder.py](src/hssm/rl/likelihoods/builder.py) (`make_rl_logp_func`, `make_rl_logp_op`):

1. **`make_addm_logp_func()`** — returns a callable `logp(data, *args)`:
   - `data[:, 0]` = rt, `data[:, 1]` = response.
   - `args` are sampled parameters in `list_params` order: `eta, kappa, a, b, x0`.
   - Extra fields `r1, r2, sacc_array, d, flag, sigma` are appended to `args` by HSSM's extra-fields machinery (see [data_validator.py:156](src/hssm/data_validator.py#L156)).
   - Body: call the vendored `_build_addm_mu_array_data(eta, kappa, r1, r2, flag_data=flag, d_data=d, max_d=sacc_array.shape[1])` to get the padded drift array, then `compute_addm_loglikelihoods(rt_data, choice_data, eta, kappa, a, b, x0, r1_data, r2_data, flag_data, sacc_array_data, d_data, ...)` from the vendored `hssm.addm.likelihoods.jax.batch`.

2. **`make_addm_logp_op()`** — wraps the JAX logp as a PyTensor `Op` with VJP, using the same pattern as `make_rl_logp_op`. The resulting `Op` gives NUTS gradients for free.

**Import contract:** `builder.py` imports the vendored likelihood as
```python
from .jax import compute_addm_loglikelihoods, _build_addm_mu_array_data
```
No other module imports the vendored subpackage.

**Reuses:**
- `make_likelihood_callable` ([distribution_utils/dist.py:718](src/hssm/distribution_utils/dist.py)) for PyTensor wrapping conventions.
- `apply_param_bounds_to_loglik` ([distribution_utils/dist.py:40-79](src/hssm/distribution_utils/dist.py)) for parameter-bound enforcement.

The `Op` returned by `make_addm_logp_op()` is **shape-polymorphic** — `compute_addm_loglikelihoods` uses `lax.scan` over trials and JAX handles dynamic `n_trials`/`max_d` through jit shape specialization. This is crucial because the `Op` is built once at registration time (commit 3) and reused for any aDDM dataset.

### Commit 3 — `feat(addm): register addm as a built-in HSSM model`

This is the central commit of the plan — it makes `hssm.HSSM(model="addm", ...)` resolve.

**File 1: `src/hssm/_types.py`** — add `"addm"` to the `SupportedModels` Literal:
```python
SupportedModels = Literal[
    "ddm", "ddm_sdv", "full_ddm", "angle", "levy", "ornstein", "weibull",
    "race_no_bias_angle_4", "ddm_seq2_no_bias", "lba3", "lba2",
    "racing_diffusion_3", "poisson_race",
    "softmax_inv_temperature_2", "softmax_inv_temperature_3",
    "addm",   # <-- new
]
```

**File 2: `src/hssm/modelconfig/addm_config.py`** (new) — pattern matches [angle_config.py](src/hssm/modelconfig/angle_config.py):
```python
from .._types import DefaultConfig
from ..addm.likelihoods.builder import make_addm_logp_op


# Build the JAX Op once at module-import time. It is shape-polymorphic
# (lax.scan over the trial batch), so the same Op handles any dataset.
_ADDM_LOGLIK_OP = make_addm_logp_op(
    list_params=["eta", "kappa", "a", "b", "z", "t"],
    extra_fields=["r1", "r2", "sacc_array", "d", "flag", "sigma"],
)


def get_addm_config() -> DefaultConfig:
    """Default configuration for the attentional drift diffusion model."""
    return {
        "response": ["rt", "response"],
        "list_params": ["eta", "kappa", "a", "b", "z", "t"],
        "choices": [-1, 1],
        "description": (
            "Attentional drift-diffusion model (Krajbich et al.). "
            "Drift rate is modulated by which option the subject is "
            "currently fixating; per-trial fixation onsets and stage counts "
            "are passed as extra_fields."
        ),
        "likelihoods": {
            "approx_differentiable": {
                "loglik": _ADDM_LOGLIK_OP,
                "backend": "jax",
                "default_priors": {},
                "bounds": {
                    "eta": (0.0, 1.0),
                    "kappa": (0.0, 5.0),
                    "a": (0.1, 3.0),
                    "b": (0.0, 1.0),
                    "z": (-1.0, 1.0),
                    "t": (0, 0.4),
                },
                "extra_fields": ["r1", "r2", "sacc_array", "d", "flag", "sigma"],
            },
        },
    }
```

**File 3: `src/hssm/defaults.py`** — the existing dispatcher `default_model_config["ddm"] = get_default_model_config("ddm")` resolves new models lazily through [`get_default_model_config`](src/hssm/modelconfig/__init__.py#L38), which imports `<name>_config` dynamically. Since we added `"addm"` to `SupportedModels`, no edit to `defaults.py` is required beyond confirming the lazy path works for `addm`. (A smoke test in commit 6 covers this.)

**File 4: `src/hssm/data_validator.py`** — small hook for aDDM-specific column-shape validation, gated on model name:
```python
def _validate_addm_columns(self):
    """Validate aDDM-specific column shapes (sacc_array is 2-D)."""
    if getattr(self.model_config, "model_name", None) != "addm":
        return
    # check r1, r2, d, flag are 1-D numeric of length n_trials
    # check sacc_array is 2-D of shape (n_trials, max_d) (or column of
    #   variable-length lists, padded inline)
    # check d[i] <= sacc_array.shape[1]
    # check flag in {0, 1}
    ...
```
Call this from the existing `_post_check_data_sanity` (the validator's post-check hook). For any non-aDDM model it is a no-op, so the change is non-intrusive.

> **Note on layering:** this is one of two places where aDDM-specific code touches generic HSSM modules. The other (commit 3) is the one-line addition to `SupportedModels`. Everything else lives under `src/hssm/addm/` or `src/hssm/modelconfig/addm_config.py`.

### Commit 4 — `test(addm): unit tests for builder, likelihood parity`

Create `tests/addm/`:

- `tests/addm/test_addm_builder_output_shape.py` — patterned after [tests/test_rl_builder_output_shape.py](tests/test_rl_builder_output_shape.py). Tiny synthetic batch; assert `make_addm_logp_func` output shape is `(n_trials,)` and dtype matches `get_jax_dtype()`.
- `tests/addm/test_addm_likelihood.py` — patterned after [tests/test_rldm_likelihood.py](tests/test_rldm_likelihood.py). **Parity check:** for a 10-trial fixture, the registered `Op` returns the same value (within 1e-6) as a direct call to the vendored `compute_addm_loglikelihoods`, and matches `compute_addm_logfptd` summed over trials. Also assert gradients w.r.t. each parameter are finite.
- `tests/addm/test_addm_registration.py` — **new**, no RLSSM analog. Assertions:
  1. `get_default_model_config("addm")` returns a `DefaultConfig` with the expected keys.
  2. `"addm" in get_args(SupportedModels)`.
  3. `hssm.HSSM(model="addm", data=tiny_df)` constructs without error.
  4. `hssm.show_defaults("addm", "approx_differentiable")` (see [`defaults.py:68`](src/hssm/defaults.py#L68)) returns a non-empty string.

**Reuses:** test fixtures from `tests/conftest.py`. Synthetic trial generation: use `efficient_fpt.aDDModel(...).generate_experiment(...)` (dev dependency at test-time only) to produce a small dataset, then materialize as a pandas DataFrame.

### Commit 5 — `test(addm): end-to-end smoke and parameter recovery`

- `tests/addm/test_addm.py` — patterned after [tests/test_rlssm.py](tests/test_rlssm.py). 200-trial synthetic dataset:
  ```python
  model = hssm.HSSM(model="addm", data=trials).sample(draws=5, tune=5)
  ```
  Asserts the call completes and returns an `InferenceData`.
- `tests/scripts/addm_recovery.py` — off-CI script; 1000-trial recovery with known `(eta, kappa, a, b, x0, sigma)`, fit via `hssm.HSSM(model="addm", data=...)`, confirm posterior means within ~2σ of ground truth. Reuse the recovery setup from [efficient-fpt examples/example8_empirical](data/azhang/efficient-fpt/examples/example8_empirical) where applicable.

### Commit 6 — `feat(addm): sampled non-decision time as pre-decision motor delay`

Adds `t` to `list_params` and shifts both `rt` and `sacc_array` by `t` before the JAX kernel runs. **Interpretation:** `t` is a pre-decision motor/encoding delay — during `[0, t]` neither the decision diffusion nor the recorded fixations contribute to drift evolution, and at decision-time 0 the subject is in whichever fixation stage contains real-time `t`. 
**Why this is more than "subtract `t` from `rt`":** `sacc_array` is recorded in *trial-time* coordinates (with `sacc_array[0] = 0`) while the decision-only likelihood expects *decision-time* coordinates. A naive `rt - t` shift without touching `sacc_array` would mis-align fixation stage onsets relative to the decision process. The full transformation is:

| Quantity | Trial-time (input) | Decision-time (passed to kernel) |
|---|---|---|
| Response time | `rt` | `rt - t` |
| Stage 0 onset | `sacc_array[0] = 0` | `0` *(kept anchored to decision start)* |
| Stage `i >= 1` onset | `sacc_array[i]` | `sacc_array[i] - t` |
| Stage count | `d` | `d` *(unchanged — see constraint below)* |
| First-fixation flag | `flag` | `flag` *(unchanged — see constraint below)* |

**Critical constraint — `t` lives entirely inside the first fixation.** That is, `t < sacc_array[i, 1]` for every trial. With this constraint:
- The first stage's effective decision-time duration shrinks from `sacc_array[1]` to `sacc_array[1] - t` (truncated head).
- All subsequent stage durations (`sacc_array[i+1] - sacc_array[i]` for `i >= 1`) are preserved.
- No stage is dropped, so the first-fixation `flag` does not flip (it would flip if we ever needed to skip an odd number of pre-decision stages).
- No piecewise re-bucketing inside JAX — the transformation is a smooth function of `t`, so NUTS gets clean gradients everywhere except the boundary `t = sacc_array[i, 1]`, which is bounded out of the support.

This constraint is consistent with the empirical literature: motor/encoding NDT is typically ~150–400 ms, while first fixations are typically 500 ms+. The configurable bound enforces it (and the parity test in commit 4 should be extended to confirm it on the recovery fixture).

**Code change — `src/hssm/addm/likelihoods/builder.py`:**

```python
def make_addm_logp_func():
    def logp(data, eta, kappa, sigma, a, b, x0, t,
             r1, r2, sacc_array, d, flag):
        rt = data[:, 0]
        response = data[:, 1]

        rt_shifted = rt - t

        # Shift stage onsets but keep index 0 anchored at 0.
        # JAX requires an out-of-place update.
        sacc_shifted = sacc_array.at[:, 1:].add(-t)

        # Validity mask:
        #   (a) decision time positive
        #   (b) t fits inside the first fixation, per the constraint above
        first_sacc_end = sacc_array[:, 1]  # max_d >= 2 guaranteed by validator
        valid = (rt_shifted > 0.0) & (t < first_sacc_end)

        ll = compute_addm_loglikelihoods(
            rt_shifted, response,
            eta, kappa, sigma, a, b, x0,
            r1, r2, flag, sacc_shifted, d,
        )
        return jnp.where(valid, ll, -jnp.inf)
    return logp
```

**Caveats baked into the implementation:**

1. **Boundary `t = 0` is well-defined** — `rt_shifted = rt`, `sacc_shifted = sacc_array`, and the kernel sees exactly the same arrays as the no-NDT path. So this commit is strictly a superset of commits 1–3.
2. **`-inf` masking, not error** — when a proposed `t` violates the constraint on any trial, we return `-inf` for that trial rather than raising. NUTS handles `-inf` cleanly (the proposal gets rejected); raising would crash the sampler.
3. **No `lax.cond` on `t`** — `jnp.where` over the full computation rather than gating with `lax.cond` keeps the trace shape-stable and avoids recompilation. The cost is computing `compute_addm_loglikelihoods` on invalid trials too, but they're masked out at the end.
4. **`max_d >= 2` requirement** — the validator (commit 3) is extended to reject datasets where any trial has `d < 2`, so `sacc_array[:, 1]` is always meaningful. Single-stage trials (whole RT spent fixating one item, no saccade) are a degenerate case that the pre-decision NDT interpretation doesn't cleanly support; reject them.
5. **Gradient through `.at[:, 1:].add(-t)`** — JAX's scatter-add propagates gradient w.r.t. `t` correctly. This is the only place `t` enters the kernel call's input arrays directly (besides `rt_shifted`), and both are differentiable scalar shifts.
6. **`set_jax_precision`** — the shifted RTs and sacc onsets can have small magnitudes (`rt - t` near 0, `sacc[i] - t` for early `i`). Recommend running this commit's tests with `set_jax_precision("x64")` to catch any roundoff sensitivity in the boundary-handling code; document that in the test file.

**Config change — `src/hssm/modelconfig/addm_config.py`:**

```python
"list_params": ["eta", "kappa", "sigma", "a", "b", "x0", "t"],  # t appended last
...
"bounds": {
    "eta":   (0.0, 1.0),
    "kappa": (0.0, 5.0),
    "a":     (0.1, 3.0),
    "b":     (0.0, 1.0),
    "x0":    (-1.0, 1.0),
    "t":     (0.0, 0.4),   # tight upper bound; refine per-dataset if needed
},
"extra_fields": ["r1", "r2", "sacc_array", "d", "flag", "sigma"],  # unchanged
```

`t` is appended last so existing tests that index `list_params` positionally don't break. The `(0.0, 0.4)` bound is a conservative default; users with shorter first fixations should override via `include=[{"name": "t", "prior": ..., "bounds": ...}]`.

**Validator change — `src/hssm/data_validator.py:_validate_addm_columns`:**

- Add: assert `sacc_array.shape[1] >= 2` and every trial has `d >= 2`.
- Add: warn (don't fail) if the model's default `t` upper bound exceeds `min(sacc_array[:, 1])` over the dataset — that proposal range will routinely violate the in-first-fixation constraint and waste gradient evaluations. Suggest tightening.

**New tests — `tests/addm/test_addm_ndt.py`:**

1. `test_ndt_zero_matches_no_ndt` — set `t = 0`, confirm `logp` matches the commit-3 no-NDT path bit-for-bit (both use the same kernel).
2. `test_ndt_shifts_logp_consistently` — for fixed `t > 0`, the JAX logp at `(rt, sacc_array, t)` equals a manually computed shift: `compute_addm_loglikelihoods(rt - t, ..., sacc_shifted, ...)`.
3. `test_ndt_invalid_t_returns_neginf` — propose `t > sacc_array[i, 1]` for some trial; assert that trial's logp is `-inf` while others remain finite.
4. `test_ndt_gradient_finite` — `jax.grad` w.r.t. `t` is finite at an interior valid point.
5. `test_ndt_recovery_smoke` — extend `tests/scripts/addm_recovery.py` (off-CI) to include `t` in the sampled parameters; confirm posterior mean of `t` is within ~2σ of the simulated value.

**Reuses:** the vendored `compute_addm_loglikelihoods` is unchanged — all NDT handling is in the HSSM-owned `builder.py`. No edit to `src/hssm/addm/likelihoods/jax/`.

> **Why this is a separate commit, not folded into commit 2.** Adding `t` after the no-NDT path is shown to work isolates the additional complexity (sacc-shift, validity masking, recovery test) from the core-correctness story. If the v1 path lands cleanly and the v1.1 NDT path turns out to need a more elaborate treatment (e.g., the pre-decision interpretation breaks down for some dataset), the v1 commits remain a working baseline.

### Commit 7 — `feat(addm): user-supplied covariates + fixation continuation in the cssm.addm PPC simulator` *(ssm-simulators, cross-repo)*

**Two coupled problems, one commit.** Faithful posterior predictives require simulating under the fitted model *on the dataset's own covariates*. Today `cssm.addm` (`ssm-simulators/src/cssm/addm_models.pyx`) does the opposite on both counts:

1. **No covariate passthrough.** It randomizes every trial-level latent internally — `r1`, `r2`, `flag`, and the fixation durations (`addm_models.pyx:145-155`) — and exposes no input for them. So it can only produce a *generic* aDDM predictive, never one conditioned on a specific trial's stimulus values or observed gaze. There is no saccade-array argument at all.
2. **Last-fixation freeze.** It pre-generates a *fixed* budget of `max_fixations` (default 100) durations and looks up drift via `_piecewise_drift`, which **returns the last stage's drift for any `t` beyond the final saccade** (`addm_models.pyx:58-66`). When a `(trial, sample)`'s decision outruns the available fixations, the particle is held at one fixed drift — biologically implausible (subjects keep saccading) and it distorts the RT/choice tails. Under defaults the ~60 s budget over-covers `max_t=20 s`, so it is *masked* in pure self-generation; but with a **finite user-supplied saccade prefix** (e.g. human fixation onsets, which only extend to the observed RT) a slow draw runs past the prefix and the freeze is *guaranteed*, not incidental.

These are coupled: once users can supply real (finite) saccade arrays, the freeze stops being a corner case and becomes the common failure mode. So both land together.

**Why `cssm.addm` (not an HSSM-side shim).** PPC must simulate under the *same* generative model that was fit, and the intention is for `cssm.addm` to host the core of efficient-fpt's collapsing-boundary aDDM (the two are converging). efficient-fpt's own `_run_heterog_trial` has the identical freeze (`efficient-fpt/src/efficient_fpt/cython/simulator.pyx:265`, the `while stage + 1 < d` guard) and its `simulate_fpt` already accepts a `sacc_array_data` prefix — so the target design unifies both: efficient-fpt's "accepts a prefix" + a continuation that efficient-fpt lacks. Fix in lockstep when the core is ported.

**Design — optional covariates + a spliced fixation sequence.** Add optional per-trial covariate inputs to `cssm.addm`; when `None`, fall back to the current internal sampling (full backward compatibility).

| Input (per trial, length `n_trials`) | When supplied | When `None` (default) |
|---|---|---|
| `r1`, `r2` | use as the stimulus ratings | randomize `int∈[1,5]` as today |
| `flag` | use as first-item indicator | randomize `0/1` as today |
| `sacc_array` (padded `n_trials × max_d`) + `d` | use as the **observed fixation prefix** | empty prefix → fully self-generated |

The fixation sequence actually used by the diffusion is always **prefix ++ continuation**:
- *Prefix* = the supplied observed onsets for that trial (or empty).
- *Continuation* = `Gamma(gamma_shape, gamma_scale)` draws (chosen option: parametric, not empirical), appended until cumulative time `≥ deadline_tmp = compute_deadline_tmp(max_t, deadline, t)`. Because the boundary collapses, `deadline_tmp` is finite (bounded by `a/b` and `max_t`), so coverage is guaranteed and `_piecewise_drift` can never reach "beyond the last saccade" within an active trial — **the freeze is structurally impossible in every regime.**

Continuation must preserve the alternation: stage `i ≥ d_prefix` takes the item opposite to stage `i−1`, derived from `flag` and parity, so the spliced drift array `[μ_first, μ_second, …]` stays consistent across the prefix→continuation seam.

Sketch (`addm_models.pyx`, replacing the fixed-budget block at ~lines 152-160; `prefix` is the supplied onsets for trial `k`, or `[0.0]` if none):

```python
# Start from the user-supplied observed onsets (decision-time, see NDT note),
# then keep saccading until the whole diffusion horizon is covered.
sacc_list = list(prefix) if prefix is not None else [0.0]
cum = sacc_list[-1]
while cum < deadline_tmp:
    cum += rng.gamma(gamma_shape, gamma_scale)
    sacc_list.append(cum)
sacc_np = np.asarray(sacc_list, dtype=np.float64)
d = len(sacc_np)
# mu_np built by alternating from `flag`, parity continued across the seam
```

`max_fixations` becomes a soft safety cap on *continuation* length (warn/raise under pathological params), not the coverage mechanism. Per `(trial, sample)`, the prefix is reused and the continuation is freshly drawn (a fresh gaze realization beyond the observed data).

**Coordinate consistency with NDT (commit 6).** Supplied saccade onsets are in **trial-time**; the decision simulator runs in **decision-time**. So before simulating, shift the supplied prefix by the (simulated/sampled) `t` exactly as commit 6 shifts the likelihood inputs (`sacc_array[1:] − t`, stage 0 anchored at 0), drop any prefix onset `≤ t`, and add `t` back to the simulated decision time at the end (`cssm.addm` already does `rts = t_particle + t + smooth_u`, `addm_models.pyx:207`). Continuation is generated in decision-time. This keeps the simulator and the commit-6 likelihood in the same coordinate frame.

**HSSM-side plumbing (this repo).** The PPC/simulate path must forward the dataset's `extra_fields` (`r1`, `r2`, `flag`, `sacc_array`, `d`) into `cssm.addm`. Wire `hssm.simulate_data(model="addm", data=...)` (and the posterior-predictive call) to pass observed covariates through; document that omitting them yields the generic self-generated predictive. No likelihood change. This is the same schema-alignment surface tracked in Open Question 4 — now load-bearing in both directions (covariates in, fixation sequence out).

**Emit the generated fixation sequence.** Extend the `full` return metadata with the per-`(trial, sample)` spliced `sacc_np`/`d`/`flag`, so PPC can check predicted *fixation counts* and gaze statistics, not just `rt`/`choice`. (Minimal-return path unchanged.)

**Cross-repo coordination (spine rule).** ssm-simulators change: after editing `addm_models.pyx`, rebuild the Cython extension, run `ssm-simulators` tests, and verify the output schema matches what `hssm.simulate_data` and the registered likelihood expect. Bump HSSM's ssm-simulators floor (commit 9) to the release containing the fix.

**Tests (ssm-simulators — cssm.addm suite):**
1. `test_covariates_respected` — supplied `r1`/`r2`/`flag` are used verbatim (drifts match `kappa·(r1−η·r2)` etc.), not randomized.
2. `test_supplied_prefix_used` — with a supplied saccade prefix and a fast decision, the simulated switch-times match the prefix exactly up to `d_prefix`.
3. `test_continuation_past_prefix` — short supplied prefix + slow decision; assert the sequence extends past `d_prefix`, alternates correctly across the seam, and reaches `deadline_tmp`.
4. `test_no_covariates_backward_compatible` — all covariates `None` reproduces the current self-generating behavior, distributionally (KS on RT, χ² on choice) — no regression.
5. `test_fixation_coverage_invariant` — for random params, `sacc_np[-1] ≥ deadline_tmp` for every `(trial, sample)`.
6. `test_ndt_coordinate_consistency` — a supplied trial-time prefix shifted by `t` yields the same decision-time sequence as a manual commit-6 shift.
7. `test_fixation_metadata_emitted` — `return_option='full'` includes per-sample spliced fixation sequences with sane counts.

> **Optional split.** This can land as two sub-commits if review prefers: **7a** covariate passthrough (inputs + plumbing + tests 1–2, 4), **7b** continuation + coverage guarantee (tests 3, 5–7). They're separable, but 7b is only *fully exercised* once 7a allows finite user prefixes, so the plan keeps them adjacent.

### Commit 8 — `docs(addm): tutorial notebook, README, mkdocs nav`

- `docs/tutorials/addm_tutorial.ipynb` — mirror the structure of [docs/tutorials/rlssm_tutorial.ipynb](docs/tutorials/rlssm_tutorial.ipynb):
  1. Generate a small aDDM dataset two ways: (a) `efficient_fpt.aDDModel(...).generate_experiment(...)`, and/or (b) `hssm.simulate_data(model="addm", ...)`. Show both as a tour of what's available.
  2. Build the model with **`hssm.HSSM(model="addm", data=..., include=[...])`** — the canonical registered-model path.
  3. Add a hierarchical regression on `eta` by participant to showcase what HSSM buys over raw efficient-fpt.
  4. `model.sample()` and plot posteriors via `arviz`.
- `mkdocs.yml` — add the tutorial to nav.
- `README.md` — add `addm` to the supported-models list.

### Commit 9 — `chore(addm): cleanup`

- Delete the stale `addm-andrew-dev/` scratch folder (which sits at the branch root) once the new code is working.
- Bump the ssm-simulators floor in `pyproject.toml` to the first release containing `addm` in `_modelconfig/` (look this up at PR time).
- Refresh the upstream commit hash in `src/hssm/addm/likelihoods/jax/NOTICE`.

---

## Files created (cumulative)

**HSSM-original code** (`src/hssm/addm/`, minimal — no user-facing class):
- `src/hssm/addm/__init__.py` — empty marker; nothing exported.
- `src/hssm/addm/likelihoods/__init__.py`
- `src/hssm/addm/likelihoods/builder.py` — `make_addm_logp_func`, `make_addm_logp_op`.

**Vendored from efficient-fpt `src/efficient_fpt/jax/`**:
- `src/hssm/addm/likelihoods/jax/__init__.py`
- `src/hssm/addm/likelihoods/jax/multi_stage.py`
- `src/hssm/addm/likelihoods/jax/single_stage.py`
- `src/hssm/addm/likelihoods/jax/batch.py`
- `src/hssm/addm/likelihoods/jax/addm_helpers.py`
- `src/hssm/addm/likelihoods/jax/utils.py`
- `src/hssm/addm/likelihoods/jax/_defaults.py`
- `src/hssm/addm/likelihoods/jax/NOTICE`

**Model registration** (peer of `src/hssm/modelconfig/angle_config.py`):
- `src/hssm/modelconfig/addm_config.py` — `get_addm_config()`; builds and embeds the JAX `Op`.

**Tests & docs**:
- `tests/addm/test_addm_registration.py`
- `tests/addm/test_addm_builder_output_shape.py`
- `tests/addm/test_addm_likelihood.py`
- `tests/addm/test_addm.py`
- `tests/addm/test_addm_ndt.py` *(commit 6)*
- `tests/addm/test_addm_simulator.py`
- `tests/scripts/addm_recovery.py`
- `docs/tutorials/addm_tutorial.ipynb`

## Files modified

- `src/hssm/_types.py` — add `"addm"` to the `SupportedModels` Literal.
- `src/hssm/data_validator.py` — add `_validate_addm_columns` hook gated on `model_config.model_name == "addm"`, called from `_post_check_data_sanity`. Commit 6 extends this to require `d >= 2` and warn when the `t` prior range exceeds first-fixation onset.
- `src/hssm/addm/likelihoods/builder.py` — commit 6 extends `make_addm_logp_func` to accept `t` and shift `rt`/`sacc_array`.
- `src/hssm/modelconfig/addm_config.py` — commit 6 appends `t` to `list_params` and adds `t` bounds.
- `pyproject.toml` — bump ssm-simulators floor to the release that registers `addm` (commit 9).
- `README.md`, `mkdocs.yml` — mention the new model and tutorial.

**Not modified** (and why):
- `src/hssm/hssm.py` / `src/hssm/base.py` — no subclass, no override; the registered-model flow already handles everything.
- `src/hssm/defaults.py` — `default_model_config` resolves new models lazily through [`get_default_model_config`](src/hssm/modelconfig/__init__.py#L38), so adding `"addm"` to `SupportedModels` is enough.
- `src/hssm/config.py` — no `aDDMConfig` class exists; the `DefaultConfig` dict in `addm_config.py` does everything a config class would.
- `src/hssm/__init__.py` — no new top-level symbol; `hssm.HSSM(model="addm", ...)` is the entry point.

## Key functions / utilities reused (no re-implementation)

| Purpose | Location |
|---|---|
| **Batched JAX FPT log-likelihood** | `hssm.addm.likelihoods.jax.compute_addm_loglikelihoods` *(vendored from `efficient_fpt.jax.batch`)* |
| Single-trial JAX (parity tests) | `hssm.addm.likelihoods.jax.compute_addm_logfptd` *(vendored from `efficient_fpt.jax.multi_stage`)* |
| Padded drift-array constructor | `hssm.addm.likelihoods.jax._build_addm_mu_array_data` *(vendored from `efficient_fpt.jax.addm_helpers`)* |
| JAX precision control | `hssm.addm.likelihoods.jax.set_jax_precision` |
| Model registration dispatcher | [hssm.modelconfig.get_default_model_config](src/hssm/modelconfig/__init__.py#L38) |
| Likelihood `Op` builder pattern | [hssm/rl/likelihoods/builder.py](src/hssm/rl/likelihoods/builder.py) |
| `LogLik` typedef (accepts `Op`) | [`hssm._types.LogLik`](src/hssm/_types.py#L11) |
| Extra-fields propagation into logp | [data_validator.DataValidatorMixin._update_extra_fields](src/hssm/data_validator.py#L156) |
| Param bound enforcement | [distribution_utils.dist.apply_param_bounds_to_loglik](src/hssm/distribution_utils/dist.py#L40) |
| **aDDM simulator (prior/posterior predictives)** | `cssm.addm` via ssm-simulators ([ssms/config/_modelconfig/addm.py](/users/azhan378/ssm-simulators/ssms/config/_modelconfig/addm.py)) |
| Synthetic dataset generation (tests) | `efficient_fpt.aDDModel.generate_experiment` *(dev-only import; not vendored)* |

---

## Verification

End-to-end checks, in order:

1. **Import** — `python -c "import hssm; hssm.HSSM(model='addm', data=tiny_df)"` constructs without error with HSSM's pinned JAX.
2. **Registry** — `hssm.show_defaults("addm", "approx_differentiable")` returns a non-empty description, and `"addm" in get_args(SupportedModels)`.
3. **Unit** — `pytest tests/addm/ -v` passes (commits 4–6).
4. **Likelihood parity** — `test_addm_likelihood.py` asserts the registered `Op` matches a direct call to `compute_addm_loglikelihoods` to 1e-6, *and* matches per-trial `compute_addm_logfptd` summed over a 10-trial fixture.
5. **Smoke sample** — `hssm.HSSM(model="addm", data=synthetic_trials).sample(draws=5, tune=5)` returns an `InferenceData`.
6. **NDT correctness** — `test_addm_ndt.py` (commit 6) confirms `t = 0` matches the no-NDT path bit-for-bit, that `t > 0` yields a logp consistent with manually shifting `rt`/`sacc_array`, and that `t` violating the in-first-fixation constraint returns `-inf` per offending trial.
7. **Simulator path** — `hssm.simulate_data(model="addm", theta=..., n_samples=...)` returns a DataFrame whose columns are a superset of what the registered `addm` likelihood consumes (including a `t` column, or with NDT applied client-side if `cssm.addm` doesn't model it). After commit 7, also assert: (a) **covariate passthrough** — supplying `data=` with `r1`/`r2`/`flag`/`sacc_array` produces predictives conditioned on those values (not randomized); (b) **no freeze** — simulated RTs show no spurious tail spike from fixation-exhausted trials and `sacc_np[-1] ≥ deadline_tmp` holds per sample; (c) omitting covariates still yields the generic self-generated predictive.
8. **Parameter recovery** (off-CI) — `tests/scripts/addm_recovery.py` confirms posterior means within ~2σ of ground truth on 1000 simulated trials, including `t` after commit 6.
9. **Tutorial runs clean** — `jupyter nbconvert --execute docs/tutorials/addm_tutorial.ipynb` finishes without errors.
10. **Docs build** — `mkdocs build` succeeds with the new tutorial in nav.

## Open questions

1. **`extra_fields` defaulting** — the model config sets `extra_fields=["r1","r2","sacc_array","d","flag"]`. Confirm that `HSSM.__init__` honors `extra_fields` from the model config dict without the user having to pass it again; if not, document the required user-side call.
2. **`t` upper bound** — the default `t ∈ (0.0, 0.4)` may be too loose for datasets with short first fixations. If the warning in `_validate_addm_columns` (commit 6) fires often, consider auto-tightening to `(0.0, 0.9 * min(sacc_array[:, 1]))` per-dataset, or expose a helper that suggests a bound.
3. **Single-stage trials (`d == 1`)** — rejected by commit 6's validator. If real datasets contain non-trivial fractions of single-fixation trials, we will need a fallback (e.g., fall back to a pure-DDM likelihood for those trials, or relax to post-decision NDT).
4. **ssm-simulators schema alignment (now bidirectional)** — commit 7 makes this load-bearing in both directions. *Inputs:* can `cssm.addm` accept the dataset's per-trial `r1`/`r2`/`flag`/`sacc_array`/`d` covariates in the shape `hssm.simulate_data` forwards them (padded ragged saccade arrays, trial-time vs decision-time)? *Outputs:* does the emitted spliced fixation sequence (`sacc_np`/`d`/`flag` per sample) land in the `full`-return metadata in the expected shape? Confirm both before wiring the PPC path.
5. **ssm-simulators version pin** — confirm which ssm-simulators release first contains `addm` in `_modelconfig/`, and bump HSSM's `pyproject.toml` floor accordingly (commit 9).

