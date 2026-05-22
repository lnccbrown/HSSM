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

**Important — this differs from prior drafts** of the plan, which named non-existent functions (`get_addm_fptd_jax_fast`, `pad_sacc_array_safely`). Those names are gone upstream. The actual entry points are `compute_addm_loglikelihoods` (batch) and `compute_addm_logfptd` (single-trial); see the use-case table above.

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
    list_params=["eta", "kappa", "a", "b", "x0"],
    extra_fields=["r1", "r2", "sacc_array", "d", "flag", "sigma"],
)


def get_addm_config() -> DefaultConfig:
    """Default configuration for the attentional drift diffusion model."""
    return {
        "response": ["rt", "response"],
        "list_params": ["eta", "kappa", "a", "b", "x0"],
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
                    "x0": (-1.0, 1.0),
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

### Commit 6 — `test(addm): ssm-simulators integration smoke`

- `tests/addm/test_addm_simulator.py` — confirm that `hssm.simulate_data(model="addm", theta=..., n_samples=...)` (which dispatches into ssm-simulators' `cssm.addm`) returns a DataFrame whose columns are a superset of what the registered `addm` model consumes. This is a sanity check that the ssm-simulators registration is wired correctly end-to-end and that HSSM's prior/posterior predictive surface works for aDDM.
- If this fails (e.g., ssm-simulators' aDDM doesn't emit `sacc_array`/`d`/`flag` in a DataFrame-friendly shape), record the gap in the commit message and open a follow-up to harmonize the schema. **Do not** block the rest of the plan on this; the JAX likelihood path is independent.

### Commit 7 — `docs(addm): tutorial notebook, README, mkdocs nav`

- `docs/tutorials/addm_tutorial.ipynb` — mirror the structure of [docs/tutorials/rlssm_tutorial.ipynb](docs/tutorials/rlssm_tutorial.ipynb):
  1. Generate a small aDDM dataset two ways: (a) `efficient_fpt.aDDModel(...).generate_experiment(...)`, and/or (b) `hssm.simulate_data(model="addm", ...)`. Show both as a tour of what's available.
  2. Build the model with **`hssm.HSSM(model="addm", data=..., include=[...])`** — the canonical registered-model path.
  3. Add a hierarchical regression on `eta` by participant to showcase what HSSM buys over raw efficient-fpt.
  4. `model.sample()` and plot posteriors via `arviz`.
- `mkdocs.yml` — add the tutorial to nav.
- `README.md` — add `addm` to the supported-models list.

### Commit 8 — `chore(addm): cleanup`

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
- `tests/addm/test_addm_simulator.py`
- `tests/scripts/addm_recovery.py`
- `docs/tutorials/addm_tutorial.ipynb`

## Files modified

- `src/hssm/_types.py` — add `"addm"` to the `SupportedModels` Literal.
- `src/hssm/data_validator.py` — add `_validate_addm_columns` hook gated on `model_config.model_name == "addm"`, called from `_post_check_data_sanity`.
- `pyproject.toml` — bump ssm-simulators floor to the release that registers `addm` (commit 8).
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
6. **Simulator path** — `hssm.simulate_data(model="addm", theta=..., n_samples=...)` returns a DataFrame whose columns are a superset of what the registered `addm` likelihood consumes.
7. **Parameter recovery** (off-CI) — `tests/scripts/addm_recovery.py` confirms posterior means within ~2σ of ground truth on 1000 simulated trials.
8. **Tutorial runs clean** — `jupyter nbconvert --execute docs/tutorials/addm_tutorial.ipynb` finishes without errors.
9. **Docs build** — `mkdocs build` succeeds with the new tutorial in nav.

## Open questions

1. **Non-decision time `t`** — include in v1 as an additional sampled parameter (shift RTs), or defer to a follow-up commit? Affects the `list_params` and `bounds` keys in `addm_config.py`.
2. **`extra_fields` defaulting** — the model config sets `extra_fields=["r1","r2","sacc_array","d","flag"]`. Confirm that `HSSM.__init__` honors `extra_fields` from the model config dict without the user having to pass it again; if not, document the required user-side call.
3. **ssm-simulators schema alignment** — does `cssm.addm`'s output via `hssm.simulate_data` emit `sacc_array`/`d`/`flag` in a DataFrame shape that matches what the registered likelihood expects? If not, commit 6 will surface the gap.
4. **ssm-simulators version pin** — confirm which ssm-simulators release first contains `addm` in `_modelconfig/`, and bump HSSM's `pyproject.toml` floor accordingly (commit 8).
