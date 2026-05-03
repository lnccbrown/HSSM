# Plan: Integrating aDDM into HSSM

## Context

The attentional drift diffusion model (aDDM; Krajbich et al.) extends the standard DDM by modulating the drift rate based on **which option the subject is currently fixating**. A fast, differentiable JAX likelihood for the aDDM has been prototyped in the sibling repo [efficient-fpt](data/azhang/efficient-fpt) — specifically `get_addm_fptd_jax_fast` in [src/efficient_fpt_jax/multi_stage.py](data/azhang/efficient-fpt/src/efficient_fpt_jax/multi_stage.py).

**Integration approach: vendor, do not depend.** Rather than add `efficient-fpt` as a dependency (it is not on PyPI, it ships compiled Cython for the simulator path that HSSM does not need, and bringing it in pulls a heavy build chain into HSSM's install), we **copy the relevant pure-JAX modules into HSSM** and own them going forward. HSSM already depends on `jax`/`jaxlib`, so the vendored code adds zero new transitive dependencies. The simulator (Cython) and NumPy/CPU paths from efficient-fpt are *not* vendored; only the `efficient_fpt_jax` subpackage's likelihood code.

The goal is to wire the aDDM into HSSM so that users can write:

```python
model = hssm.HSSM(model="addm", data=addm_trial_df, include=[...])
model.sample()
```

where `addm_trial_df` contains standard columns (`rt`, `response`) plus **aDDM-specific per-trial arrays** (item values, fixation onsets, fixation counts, first-fixation flag). The aDDM needs per-trial covariates that are *not* themselves sampled parameters — exactly the pattern RLSSM already solves in HSSM. We therefore follow the RLSSM design so that aDDM lives alongside it rather than carving a new architectural lane.

The intended outcome is a working `model="addm"` path inside HSSM that (a) validates aDDM-specific trial data, (b) composes the vendored JAX FPT likelihood with sampled parameters `{eta, kappa, sigma, a, b, x0, t}` (non-decision time optional), (c) exposes the standard HSSM hierarchical regression and sampling machinery, and (d) ships with a tutorial notebook and unit tests.

## Design choice: config pattern, not subclass

The plan creates an **`aDDMConfig`** dataclass plus a small submodule — no new `aDDM(HSSM)` subclass is introduced. (If the user prefers an explicit subclass for API discoverability, a thin `class aDDM(HSSM)` wrapper can be added on top.)

---

## Step-by-step plan

### Step 1 — Vendor the JAX likelihood code into HSSM

**Source files to copy (from efficient-fpt):**

| Source (efficient-fpt) | Destination (HSSM) | Purpose |
|---|---|---|
| `src/efficient_fpt_jax/multi_stage.py` | `src/hssm/addm/likelihoods/jax/multi_stage.py` | `get_addm_fptd_jax_fast`, `pad_sacc_array_safely` |
| `src/efficient_fpt_jax/single_stage.py` | `src/hssm/addm/likelihoods/jax/single_stage.py` | `fptd_single_jax`, `q_single_jax` (called by multi_stage) |
| `src/efficient_fpt_jax/utils.py` | `src/hssm/addm/likelihoods/jax/utils.py` | `GAUSS_LEGENDRE_30_X`, `GAUSS_LEGENDRE_30_W` quadrature constants |

**Action:**

1. Copy the three files above verbatim into a new `src/hssm/addm/likelihoods/jax/` package.
2. Update relative imports inside the copied files so they resolve within `hssm.addm.likelihoods.jax` (e.g. `from .single_stage import ...`, `from .utils import ...` — these are already relative in the source, so no change needed in practice; verify).
3. Create `src/hssm/addm/likelihoods/jax/__init__.py` exposing only the symbols HSSM needs:
   ```python
   from .multi_stage import get_addm_fptd_jax_fast, pad_sacc_array_safely
   from .single_stage import fptd_single_jax, q_single_jax
   from .utils import GAUSS_LEGENDRE_30_X, GAUSS_LEGENDRE_30_W
   ```
4. Add a header comment to each vendored file recording the upstream commit hash from `efficient-fpt` so future maintainers can diff against upstream when bug fixes land there.
5. **Do not** vendor `efficient_fpt_jax/batch.py` (not used by the per-trial likelihood we wrap), nor anything from the `efficient_fpt` (Cython/NumPy) subpackage — that path is the simulator and is not part of inference.
6. **Do not** add `efficient-fpt` to `pyproject.toml`. HSSM already depends on `jax`/`jaxlib`, so the vendored code introduces no new dependencies.

**License/attribution:** efficient-fpt ships under a permissive license (see `efficient-fpt/LICENSE`); copy that license text into `src/hssm/addm/likelihoods/jax/LICENSE` (or `NOTICE`) so attribution travels with the code. If HSSM and efficient-fpt share authors/license, a brief `# Adapted from efficient-fpt (commit <sha>)` header is sufficient.

**Rationale:** efficient-fpt is not on PyPI; its Cython compile chain is heavyweight and irrelevant to HSSM (HSSM only needs the inference likelihood, not the simulator); and pinning to a remote git dep would couple HSSM's CI to an unstable upstream. Vendoring lets HSSM ship a frozen, audited copy that evolves on HSSM's release cadence.

**Drift management:** efficient-fpt continues to be the research home for the likelihood. When upstream changes, the vendored copy can be re-synced by re-copying the three files and rebuilding tests. The upstream-commit header in step 4 makes "what version are we on?" trivially answerable. (This is the same pattern HSSM already uses for `bayesflow` from a dev branch, just snapshotted instead of git-tracked.)

### Step 2 — Define the aDDM submodule layout

Create a new package under `src/hssm/addm/`, mirroring `src/hssm/rl/`:

```
src/hssm/addm/
  __init__.py
  likelihoods/
    __init__.py
    builder.py           # make_addm_logp_func / make_addm_logp_op
    addm_jax.py          # thin wrapper that imports from .jax and applies the attention process
    jax/                 # vendored from efficient_fpt_jax (Step 1)
      __init__.py
      multi_stage.py
      single_stage.py
      utils.py
  attention_process.py   # pluggable fixation/attention models
```

**Rationale:** aDDM is conceptually a two-stage model (attention process → SSM likelihood) just like RLSSM (learning process → SSM likelihood). Reusing the folder layout makes the parallel obvious to future maintainers. The vendored JAX code lives in its own `jax/` subdirectory so it stays clearly identifiable as upstream-derived, isolated from HSSM-original code in `builder.py` and `addm_jax.py`.

### Step 3 — Add `aDDMConfig` dataclass in `config.py`

**Critical file:** [src/hssm/config.py](data/azhang/HSSM/src/hssm/config.py) — add a new dataclass beneath `RLSSMConfig` (around line 457).

```python
@dataclass
class aDDMConfig(BaseModelConfig):
    """Config for the attentional DDM."""
    model_name: str = "addm"
    list_params: list[str] = field(
        default_factory=lambda: ["eta", "kappa", "sigma", "a", "b", "x0"]
    )
    params_default: list[float] = field(
        default_factory=lambda: [0.3, 1.0, 1.0, 2.0, 0.0, 0.0]
    )
    response: list[str] = field(default_factory=lambda: ["rt", "response"])
    choices: tuple[int, ...] = (-1, 1)
    # trial-level covariates consumed by the attention process:
    extra_fields: list[str] | None = field(
        default_factory=lambda: ["r1", "r2", "sacc_array", "d", "flag"]
    )
    bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    loglik_kind: str = "approx_differentiable"
    attention_process: str | Callable = "standard_alternating"
    description: str | None = "Attentional Drift Diffusion Model"

    def to_config(self) -> Config: ...
```

**Key design decisions:**
- `extra_fields` defaults to the five aDDM-specific columns that the JAX likelihood needs. These are **not** sampled parameters — they come from the data.
- `attention_process` is a pluggable hook (default `"standard_alternating"`) that maps `(r1, r2, flag, eta, kappa) → mu_array_padded` per trial. This mirrors `RLSSMConfig.learning_process`.
- `list_params` covers the sampled parameters. Non-decision time `t` is deliberately omitted initially; it can be added later via shifted RTs.
- `.to_config()` builds a standard HSSM `Config` pointing at the new likelihood op from Step 4, so downstream `HSSM.__init__` behavior is unchanged.

**Reuses:** `BaseModelConfig` (config.py:48), `Config.from_defaults` registration flow (config.py:96–145), `register_model` (register.py:16–60).

### Step 4 — Build the likelihood op in `addm/likelihoods/builder.py`

Mirror [hssm/rl/likelihoods/builder.py](data/azhang/HSSM/src/hssm/rl/likelihoods/builder.py) (`make_rl_logp_func`, `make_rl_logp_op`).

Two functions:

1. **`make_addm_logp_func(attention_process)`** — returns a callable `logp(data, *args)` where:
   - `data[:, 0]` = rt, `data[:, 1]` = response
   - `args` are sampled parameters in `list_params` order: `eta, kappa, sigma, a, b, x0`
   - extra fields `r1, r2, sacc_array, d, flag` are appended to `args` by the HSSM extra-field machinery (exactly as RLSSM does; see [data_validator.py:156](data/azhang/HSSM/src/hssm/data_validator.py#L156)).
   - Internally: call the attention process to build `mu_array_padded`, then call `get_addm_fptd_jax_fast(t=rt, d=d, mu_array=mu_array, sacc_array=sacc_array, sigma, a, b, x0)` vmapped over trials.

2. **`make_addm_logp_op(attention_process)`** — wraps the JAX logp as a PyTensor `Op` with VJP, using the same pattern as `make_rl_logp_op`. This gives NUTS gradients for free.

**Reuses:**
- `get_addm_fptd_jax_fast` and `pad_sacc_array_safely` — imported from the vendored `hssm.addm.likelihoods.jax` package (Step 1).
- `make_likelihood_callable` ([distribution_utils/dist.py:718](data/azhang/HSSM/src/hssm/distribution_utils/dist.py)) for PyTensor wrapping conventions.
- `apply_param_bounds_to_loglik` ([distribution_utils/dist.py:40–79](data/azhang/HSSM/src/hssm/distribution_utils/dist.py)) for parameter-bound enforcement.

**Import contract:** `builder.py` imports the vendored likelihood as `from hssm.addm.likelihoods.jax import get_addm_fptd_jax_fast, pad_sacc_array_safely`. No code outside `hssm.addm` should import from the vendored `jax/` subpackage directly — keeping the import surface narrow makes a future re-vendor (or replacement with an upstream PyPI release, should one ever appear) a single-file change.

### Step 5 — Attention process ("learning process" analog)

File: `src/hssm/addm/attention_process.py`.

Default implementation `standard_alternating(r1, r2, flag, eta, kappa, max_d) -> mu_padded`:

```
mu1 = kappa * (r1 - eta * r2)
mu2 = kappa * (eta * r1 - r2)
# alternate mu1/mu2 by stage parity, respecting `flag` for first fixation
# return shape (n_trials, max_d)
```

This reproduces the logic in [efficient-fpt addm.py:_build_mu_data_padded](data/azhang/efficient-fpt/src/efficient_fpt/addm.py) but in JAX for autodiff. Expose it via a registry so future variants (e.g., bias, drift offsets) can be registered by name — the same way `RLSSMConfig.learning_process` accepts either a string or dict.

### Step 6 — Register `"addm"` as a built-in model

**Critical files:**
- [src/hssm/modelconfig/](data/azhang/HSSM/src/hssm/modelconfig/) — add `addm_config.py` in the same style as the existing per-model configs (e.g., `ddm_config.py`).
- [src/hssm/defaults.py](data/azhang/HSSM/src/hssm/defaults.py) — register `"addm"` in the default model list so `hssm.HSSM(model="addm", ...)` works out of the box.

`addm_config.py` returns a dict with `response`, `list_params`, `choices`, `description`, and a `likelihoods` sub-dict keyed `"approx_differentiable"` whose `loglik` points to `make_addm_logp_op(...)` from Step 4 and `extra_fields=["r1","r2","sacc_array","d","flag"]`.

**Reuses:** `register_model` (register.py:16–60) — already handles the registration flow; we just need to pass the right dict.

### Step 7 — Data validation for aDDM-specific columns

**Critical file:** [src/hssm/data_validator.py](data/azhang/HSSM/src/hssm/data_validator.py).

The DataValidatorMixin currently validates that `extra_fields` columns exist ([line 46](data/azhang/HSSM/src/hssm/data_validator.py#L46)). aDDM also needs **shape validation** because `sacc_array` is a 2D array-of-arrays stored inside a DataFrame column.

Add an optional `_validate_addm_columns()` method invoked when `model_config.model_name == "addm"`:
- `r1`, `r2`, `d`, `flag` must be 1D numeric with length `n_trials`.
- `sacc_array` must be a 2D array of shape `(n_trials, max_d)` (or a column of variable-length lists, padded internally via `pad_sacc_array_safely`).
- `d[i] <= sacc_array.shape[1]`.

Minimally invasive: put the hook in `_post_check_data_sanity` and no-op for non-aDDM models.

### Step 8 — Tests

New file: `tests/test_addm_config.py`, patterned after [tests/test_rlssm_config.py](data/azhang/HSSM/tests/test_rlssm_config.py):

1. `TestaDDMConfigCreation` — build `aDDMConfig`, assert defaults.
2. `TestaDDMConfigConversion` — `.to_config()` round-trip.
3. `TestaDDMLikelihood` — tiny synthetic dataset (10 trials), confirm `logp` is finite, gradient w.r.t. each parameter is finite, matches a direct call to `get_addm_fptd_jax_fast`.
4. `TestaDDMEndToEnd` — 200-trial synthetic dataset, `hssm.HSSM(model="addm", ...)` builds, a single MCMC draw succeeds (smoke test, `draws=5, tune=5`).

**Reuse:** test fixtures from `tests/conftest.py`.

### Step 9 — Tutorial notebook

Create `docs/tutorials/addm_tutorial.ipynb` mirroring the structure of `docs/tutorials/rlssm_tutorial.ipynb`:
- Load/simulate a small aDDM dataset (reuse `simulate_addm` from efficient-fpt example6).
- Build the HSSM model with `model="addm"`.
- Add a hierarchical regression on `eta` (e.g., by participant) to showcase why using HSSM buys more than raw efficient-fpt.
- Run `model.sample()` and plot posteriors via `arviz`.

### Step 10 — Cleanup

- Delete or rename the stale [addm_andrew_dev](data/azhang/HSSM/addm_andrew_dev) folder (it has a trailing space in its name, which is a foot-gun on many filesystems) once the new module is working.
- Update `README.md` example list and `mkdocs.yml` nav to include the new tutorial.

---

## Files to be created

**HSSM-original code:**
- `src/hssm/addm/__init__.py`
- `src/hssm/addm/attention_process.py`
- `src/hssm/addm/likelihoods/__init__.py`
- `src/hssm/addm/likelihoods/builder.py`
- `src/hssm/addm/likelihoods/addm_jax.py`
- `src/hssm/modelconfig/addm_config.py`
- `tests/test_addm_config.py`
- `docs/tutorials/addm_tutorial.ipynb`

**Vendored from efficient-fpt (verbatim copies, kept in their own subpackage):**
- `src/hssm/addm/likelihoods/jax/__init__.py`
- `src/hssm/addm/likelihoods/jax/multi_stage.py` ← `efficient_fpt_jax/multi_stage.py`
- `src/hssm/addm/likelihoods/jax/single_stage.py` ← `efficient_fpt_jax/single_stage.py`
- `src/hssm/addm/likelihoods/jax/utils.py` ← `efficient_fpt_jax/utils.py`
- `src/hssm/addm/likelihoods/jax/NOTICE` (or LICENSE) — upstream attribution and commit hash

## Files to be modified

- `src/hssm/config.py` — add `aDDMConfig` dataclass.
- `src/hssm/defaults.py` — register `"addm"` in the default model list.
- `src/hssm/data_validator.py` — add aDDM column-shape validation hook.
- `README.md`, `mkdocs.yml` — mention the new model.

(`pyproject.toml` is **not** modified — no new dependencies are introduced. JAX is already a core dependency.)

## Key functions/utilities to reuse (no re-implementation)

| Purpose | Location |
|---|---|
| JAX FPT likelihood | `hssm.addm.likelihoods.jax.get_addm_fptd_jax_fast` *(vendored)* |
| Safe padding of saccade arrays | `hssm.addm.likelihoods.jax.pad_sacc_array_safely` *(vendored)* |
| Likelihood op wrapping pattern | [hssm/rl/likelihoods/builder.py](data/azhang/HSSM/src/hssm/rl/likelihoods/builder.py) |
| Config → standard Config conversion | [config.RLSSMConfig.to_config](data/azhang/HSSM/src/hssm/config.py#L408) |
| Model registration | [register.register_model](data/azhang/HSSM/src/hssm/register.py#L16) |
| Extra-fields propagation into logp | [data_validator.DataValidatorMixin._update_extra_fields](data/azhang/HSSM/src/hssm/data_validator.py#L156) |
| Param bound enforcement | [distribution_utils.dist.apply_param_bounds_to_loglik](data/azhang/HSSM/src/hssm/distribution_utils/dist.py#L40) |

---

## Verification

End-to-end checks, in order:

1. **Unit**: `pytest tests/test_addm_config.py -v` — all four test classes pass, including finite-gradient check.
2. **Likelihood parity**: in `test_addm_config.py::TestaDDMLikelihood`, assert HSSM's wrapped op returns the same value (to 1e-6) as a direct call to the vendored `hssm.addm.likelihoods.jax.get_addm_fptd_jax_fast` on a 10-trial fixture. This confirms the HSSM extra-fields/op-wrapping plumbing does not corrupt the underlying JAX computation. (A separate, off-CI sanity script may also compare against an installed `efficient-fpt` checkout to detect drift between the vendored copy and upstream.)
3. **Smoke sample**: `hssm.HSSM(model="addm", data=synthetic_trials).sample(draws=5, tune=5)` completes without error and returns an `InferenceData`.
4. **Parameter recovery**: larger off-CI script (e.g., `tests/scripts/addm_recovery.py`) — simulate 1000 trials with known `(eta, kappa, a, b, x0, sigma)`, fit in HSSM, confirm posterior means within ~2σ of ground truth. Reuse the recovery setup from [efficient-fpt example8_empirical/parameter_recovery.ipynb](data/azhang/efficient-fpt/examples/example8_empirical).
5. **Tutorial runs clean**: `jupyter nbconvert --execute docs/tutorials/addm_tutorial.ipynb` finishes without errors.
6. **Docs build**: `mkdocs build` succeeds with the new tutorial in nav.

## Open questions for the user

1. **Subclass vs config-only**: confirm the config-pattern approach (no `class aDDM(HSSM)`) is acceptable, or whether a thin `hssm.aDDM` convenience class is desired on top.
2. **Non-decision time `t`**: include in v1 as an additional sampled parameter (shift RTs), or defer?
3. **Attention-process extensibility**: is the default `standard_alternating` enough, or should v1 already expose user-pluggable attention processes (e.g., non-alternating fixation patterns)?
