# aDDM Integration into HSSM — Development Summary

This document summarizes the four-commit sequence that integrates a differentiable
attentional drift-diffusion model (aDDM) into [HSSM](https://github.com/lnccbrown/HSSM).
Each commit builds on the last: vendored kernel → likelihood Op → config → model
subclass.

---

## Commit 1 — Vendored JAX Kernel

**Goal:** Bring in the pure-JAX aDDM log-likelihood from the
[efficient-fpt](https://github.com/the-beekeepers/efficient-fpt) project without
adding a new runtime dependency or its heavyweight Cython build chain.

The kernel is *vendored* — a frozen in-tree copy — rather than imported as a
package. JAX/jaxlib are already HSSM dependencies, so this adds nothing new to
`pyproject.toml`.

**Source commit:** `d97a451479141acef845195610f0f9d85824844e` of efficient-fpt,
from `src/efficient_fpt/jax/`.

**Files copied to `src/hssm/addm/`:**

| Vendored file | Upstream source |
|---|---|
| `batch.py` | `src/efficient_fpt/jax/batch.py` |
| `multi_stage.py` | `src/efficient_fpt/jax/multi_stage.py` |
| `single_stage.py` | `src/efficient_fpt/jax/single_stage.py` |
| `addm_helpers.py` | `src/efficient_fpt/jax/addm_helpers.py` |
| `utils.py` | `src/efficient_fpt/jax/utils.py` |
| `_defaults.py` | `src/efficient_fpt/_defaults.py` |
| `_quadrature.py` | `src/efficient_fpt/quadrature.py` |

**HSSM-authored modifications** (must be re-applied if re-vendoring):

1. **Import retargeting** to make the subpackage self-contained (relative imports
   updated to match the new directory structure).
2. `resolve_quadrature_orders` folded from `utils.py` into the vendored copy
   (only that function; `adaptive_interpolation` was not copied).
3. A public `compute_addm_loglikelihoods_from_mu` wrapper added to `batch.py`
   over the private core function, providing a stable entry point that accepts a
   pre-built per-trial drift array.

> **Do not edit vendored files directly.** To update, re-vendor the whole
> directory and re-apply the modifications above.

---

## Commit 2 — Attention-Process Registry + Likelihood Op Builder

**Goal:** Turn the Commit 1 JAX kernel into a usable, differentiable PyMC
likelihood. Mirrors the architecture of `src/hssm/rl/likelihoods/builder.py` and
reuses HSSM's existing JAX→PyTensor machinery.

**Files added:**

| File | Purpose |
|---|---|
| `src/hssm/addm/attention_process.py` | Name→callable registry. `standard_alternating` is the default; `resolve_attention_process(name_or_callable)` resolves a registry key or callable. |
| `src/hssm/addm/likelihoods/builder.py` | `make_addm_logp_func` → JAX log-likelihood; `make_addm_logp_op` → differentiable PyTensor Op. |
| `src/hssm/addm/likelihoods/__init__.py` | Exports both builder functions. |

**Key design decisions:**

- **Column-ordering contract:** `loglik(data, *dist_params, *extra_fields)` with
  `list_params = [eta, kappa, a, b, x0]` and
  `extra_fields = [r1, r2, flag, sacc_array, d, sigma]`. The closure reorders
  these into the kernel's positional slots.
- **Default vs. custom path:** The default `standard_alternating` process calls
  `compute_addm_loglikelihoods` (kernel builds the drift array); any custom
  process produces `mu` and calls `compute_addm_loglikelihoods_from_mu`. The
  branch is decided once at build time (jit-friendly).
- **Gradients:** `n_params = len(list_params)` — gradients are computed only for
  sampled parameters; extra-field gradients are undefined.
- **Import-cycle fix:** `standard_alternating` imports the vendored kernel lazily
  (inside the function body) to avoid a circular import between
  `attention_process` and `likelihoods/__init__`.

**Tests (`test_commit2_builder.py`):** 5 tests, all passing — output shape,
kernel equivalence, PyTensor Op equivalence, gradient finiteness, and custom
attention-process routing.

---

## Commit 3 — `aDDMConfig` Dataclass

**Goal:** Add the config object that names sampled parameters, per-trial
covariates, bounds, and the attention process. Mirrors `RLSSMConfig` on
`BaseModelConfig`.

**File added:**

| File | Purpose |
|---|---|
| `src/hssm/addm/config.py` | `@dataclass aDDMConfig(BaseModelConfig)` with `validate()`, `get_defaults(param)`, and `from_addm_dict()`. |

**Key design decisions:**

- Every field defaults so `aDDMConfig()` produces a complete, valid config with
  no required arguments.
- `loglik` / `backend` are inherited as `None` and injected later by
  `aDDM.__init__` via `dataclasses.replace` (Commit 4) — not redeclared here.
- `validate()` checks: `list_params` is present, `attention_process` resolves,
  `params_default` length matches `list_params`, every parameter has a `bounds`
  entry, and `extra_fields` is present.


**Tests (`test_commit3_config.py`):** 8 tests, all passing — defaults, subclass
instantiation, validation success and failure cases, `get_defaults`, and
`from_addm_dict` round-trip.

---

## Commit 4 — `aDDM(HSSMBase)` Subclass + Top-Level Export

**Goal:** Wire everything together so `hssm.aDDM(data=..., model_config=aDDMConfig(...))` constructs and samples. Mirrors `RLSSM` but without the balanced-panel reshape.

**Files added/changed:**

| File | Change |
|---|---|
| `src/hssm/addm/addm.py` | NEW — `class aDDM(HSSMBase)`: `__init__`, `_make_model_distribution`, `_update_extra_fields`, data validation/materialization helpers. |
| `src/hssm/addm/__init__.py` | Exports `aDDM`, `aDDMConfig`, and the attention-process registry. |
| `src/hssm/__init__.py` | Top-level exports of `aDDM` and `aDDMConfig` (peers of `RLSSM`). |
| `src/hssm/addm/likelihoods/builder.py` | Reduces `sigma` to a scalar in `logp` (integration fix). |

**Integration fixes discovered during wiring** (not in the original plan):

1. **`sacc_array` as an object column crashes bambi.** `bmb.Model` tries to
   categorize all object columns; array cells are unhashable and fail `factorize`.
   Fix: `_prepare_addm_data` converts `sacc_array` to padded hashable tuples
   before calling `super().__init__`. `_make_model_distribution` and an overridden
   `_update_extra_fields` both reconstruct the `(n_trials, max_d)` float array
   via `_stack_sacc_array`.

2. **Core params must stay scalar.** The aDDM kernel vmaps over trials but treats
   `eta, kappa, a, b, x0` as scalars. HSSM's default broadcast would expand them
   to `(n_obs,)` arrays, causing a shape mismatch. Fix: `params_is_trialwise = all
   False`. Consequence: per-trial / regression core params are not yet supported
   by the kernel — such models build but should not be sampled.

3. **`sigma` must be scalar.** As a model-level diffusion constant, passing an
   `(n_obs,)` vector collided with the `(order,)` quadrature grid. Fix: the
   builder reduces `sigma` to a scalar; the validator requires the `sigma` column
   to be constant across all trials.

4. **Post-hoc `log_likelihood` (WAIC/LOO) unsupported.** HSSM's pointwise
   log-likelihood re-evaluates the Op with a draw dimension on params, which the
   scalar-param kernel cannot broadcast. Workaround: sample with
   `idata_kwargs={"log_likelihood": False}`. Supporting it requires the kernel to
   carry a sample dimension.

**Tests (`test_commit4_subclass.py`):** 7 tests, all passing — default and
explicit config construction, subclass checks, Op injection, bad-column
validation, smoke sampling (200 trials, 5 draws), and hierarchical regression
build.

---

## Current Limitations / Future Work

| Item | Notes |
|---|---|
| Per-trial / regression core params | Build succeeds but sampling unsupported until the kernel supports a per-trial scalar dimension. |
| Post-hoc log-likelihood (WAIC/LOO) | Requires the kernel to carry a sample dimension. |
| `missing_data` / `deadline` interaction | Forwarded to the base class; interaction with `sacc_array` materialization is untested. |
| PPC-side coordinate handling | Planned for a future commit (Commit 6 in original sketch). |
