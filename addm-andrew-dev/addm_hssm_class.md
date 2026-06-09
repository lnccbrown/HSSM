# Plan: Integrating aDDM into HSSM as an `HSSMBase` subclass

Branch: `addm-integration-class-0` (HSSM repo). Branch off `main` at current HEAD.

> **Architectural stance (read first).** aDDM is integrated as a **first-class subclass of
> `HSSMBase`** — `class aDDM(HSSMBase)` — mirroring the existing
> [`RLSSM(HSSMBase)`](src/hssm/rl/rlssm.py) at [rl/rlssm.py:41](src/hssm/rl/rlssm.py#L41).
> The user-facing entry point is `hssm.aDDM(data=..., model_config=aDDMConfig(...))`, **not**
> `hssm.HSSM(model="addm", ...)`. aDDM is therefore **not** registered in the
> `SupportedModels` literal and there is **no** `modelconfig/addm_config.py`. This is the
> deliberate counterpart to the sibling plan [addm_hssm.md](addm_hssm.md), which explores the
> *registered-model* formulation; this document keeps the **subclass formulation** and only
> imports the design ideas that are consistent across both (commit-structured delivery,
> non-decision time as a sampled parameter, a fixation-continuation stage for PPC, and
> commit-wise tests).

## Context

The attentional drift diffusion model (aDDM; Krajbich et al.) extends the standard DDM by
modulating the drift rate based on **which option the subject is currently fixating**. The
sampled parameters are `{eta, kappa, a, b, x0}` (plus optional non-decision time `t`,
added in Commit 5); the **per-trial covariates** `{r1, r2, flag, sacc_array, d, sigma}` come from the
data and are *not* sampled — exactly the pattern [`RLSSM`](src/hssm/rl/rlssm.py) already solves
with its extra-fields machinery. We therefore follow the RLSSM design point-for-point so aDDM
lives alongside it as a peer subclass rather than carving a new architectural lane.

Two upstream packages supply most of the machinery:

1. **efficient-fpt** (sibling repo) — provides a fast, differentiable **JAX batch
   log-likelihood** for the aDDM. It has been **restructured**: there is no longer a separate
   `efficient_fpt_jax` package (it still exists but is legacy); the maintained code lives under
   a unified `efficient_fpt/` with `jax/`, `numpy/`, and `cython/` backend subpackages. The
   batched JAX kernel we wrap is `efficient_fpt.jax.compute_addm_loglikelihoods` (an alias of
   `compute_addm_loglikelihoods_batchscan`).
2. **ssm-simulators** — HSSM's standard simulator dependency. It ships a Cython aDDM simulator
   `cssm.addm` (`ssm-simulators/src/cssm/addm_models.pyx`). We use it for prior/posterior
   predictives, and Commit 6 extends it (cross-repo) to accept user covariates and to keep
   saccading past the final observed fixation.

### Integration strategy: vendor the JAX likelihood, depend on ssm-simulators

- **Vendor** (copy into HSSM, no new runtime dependency): the pure-JAX likelihood code from
  `efficient_fpt/jax/`. efficient-fpt is not on PyPI, its Cython compile chain is heavyweight,
  and HSSM already depends on `jax`/`jaxlib`. Vendoring ships a frozen, audited copy that
  evolves on HSSM's release cadence. The Cython/NumPy backends and the simulator are **not**
  vendored.
- **Depend** (no copy): ssm-simulators is already a transitive dependency; `cssm.addm` gives us
  the simulator path.

### Why a subclass (and not the registered-model path)

`RLSSM` is a subclass — not a `SupportedModels` entry — because its likelihood is a **pre-built
differentiable PyTensor `Op`** that supersedes the `loglik`/`loglik_kind` dispatch used by
[`HSSM`](src/hssm/hssm.py#L82). The Op is constructed in `__init__` and injected via
`dataclasses.replace(model_config, loglik=op, backend="jax")`, after which
`_make_model_distribution()` is overridden to consume it directly
([rl/rlssm.py:209](src/hssm/rl/rlssm.py#L209)). aDDM has the identical need: a JAX Op wrapping
`compute_addm_loglikelihoods`, with VJP gradients for NUTS. The subclass is the natural home
for (a) building that Op, (b) the pluggable **attention process**, and (c) aDDM-specific data
validation — all without touching generic HSSM modules.

**One simplification over RLSSM.** RLSSM reshapes rows into a `(n_participants, n_trials)`
balanced panel because its *learning* process is sequential within participant, and it
therefore rejects `missing_data`/`deadline`. aDDM has **no within-participant sequential
dependency** — each trial's likelihood depends only on that trial's own covariates — so aDDM
needs **no panel reshaping, no balanced-panel requirement**, and can leave `missing_data`/
`deadline` to the base class. Hierarchical regressions (e.g. `eta` by participant) are handled
by HSSM's ordinary `include=[...]` machinery, not by the Op.

### efficient-fpt: what this plan consumes (and what it does not)

| Use case | Function (post-restructure) | Role here |
|---|---|---|
| **Batched likelihood for NUTS (primary)** | `efficient_fpt.jax.compute_addm_loglikelihoods` (= `..._batchscan`) | Production kernel the Op wraps. Signature: `(rt, choice, eta, kappa, sigma, a, b, x0, r1, r2, flag, sacc_array, d, *, order_mid, order_last, ...)`. Builds the drift array internally. |
| Drift-array helper (JAX) | `efficient_fpt.jax.addm_helpers._build_addm_mu_array_data` | `(eta, kappa, r1, r2, flag, d, max_d) → (n_trials, max_d)` padded drift array. This **is** the default attention process; vendored and reused. |
| Single-trial JAX (parity tests only) | `efficient_fpt.jax.compute_addm_logfptd` | Used in unit tests to confirm batched output matches per-trial output. Not on the inference path. |
| Precision control | `efficient_fpt.jax.set_jax_precision`, `get_jax_dtype` | Called explicitly in tests; importing the package does **not** mutate global JAX precision. |
| Optimizer NLL closure | `efficient_fpt.jax.make_addm_nll_function` | Reference/parity only; HSSM needs per-trial log-likes summed by PyMC, not a pre-built mean NLL. |
| **Simulator** | *Do not import from efficient-fpt* | The simulator surface goes through ssm-simulators (`cssm.addm`). |
| Synthetic data (tests) | `efficient_fpt.aDDModel.generate_experiment` | Dev-only import to fabricate fixtures; HSSM does not depend on it at runtime. |

### Parameter scheme (grounded in the kernel signature)

| Kind | Names | Notes |
|---|---|---|
| **Sampled** (`list_params`) | `eta, kappa, a, b, x0` → `+ t` (Commit 5) | `kappa` = drift scaling, `eta` = attentional discount, `a` = boundary, `b` = collapse rate, `x0` = start point, `t` = non-decision time.  |
| **Per-trial covariates** (`extra_fields`) | `r1, r2, flag, sacc_array, d, sigma` | Ordered to match the kernel's positional covariate slots. Not sampled. |

`t` is appended **last** to `list_params` so positional indexing in earlier tests doesn't break,
and it is consumed **inside the builder** (it shifts `rt`/`sacc_array`); it is *not* a kernel
argument.

---

## Plan, organized by commits

Each commit is self-contained, reviewable, and passes its own tests. Every commit carries a
**Tests** subsection asserting that commit's functionality in isolation.

### Commit 1 — `vendor: copy efficient-fpt JAX likelihood into hssm.addm.likelihoods.jax`

Mirror the way HSSM snapshots dev-branch dependencies, but as a frozen in-tree copy.

**Source → destination (verbatim copies from `efficient_fpt/jax/`):**

| Source (efficient-fpt) | Destination (HSSM) |
|---|---|
| `efficient_fpt/jax/batch.py` | `src/hssm/addm/likelihoods/jax/batch.py` |
| `efficient_fpt/jax/multi_stage.py` | `src/hssm/addm/likelihoods/jax/multi_stage.py` |
| `efficient_fpt/jax/single_stage.py` | `src/hssm/addm/likelihoods/jax/single_stage.py` |
| `efficient_fpt/jax/addm_helpers.py` | `src/hssm/addm/likelihoods/jax/addm_helpers.py` |
| `efficient_fpt/jax/utils.py` | `src/hssm/addm/likelihoods/jax/utils.py` |
| `efficient_fpt/_defaults.py` (constants used by `batch`/`multi_stage`) | `src/hssm/addm/likelihoods/jax/_defaults.py` |

**Import retargeting (this is more than the old 3-file copy — verified against the source tree):**
the `jax/` files reach *above* the subpackage in three places, all of which must be repointed:

- `batch.py` and `multi_stage.py`: `from .._defaults import (...)` → `from ._defaults import (...)`
  (now a sibling).
- `batch.py` and `multi_stage.py`: `from ..utils import resolve_quadrature_orders` → **inline**
  `resolve_quadrature_orders` into the vendored `jax/utils.py` (small helper) and import it
  locally.
- `utils.py`: `from ..quadrature import (...)` → **inline** the handful of quadrature constants/
  helpers actually referenced into `jax/utils.py` (or add `jax/_quadrature.py`). Do **not** pull
  in all of `efficient_fpt/quadrature.py`.

**Actions:**

1. Create `src/hssm/addm/likelihoods/jax/` and copy the six files; apply the retargeting above.
2. Header comment in each copied file recording the upstream commit hash at copy time
   (`# Vendored from efficient-fpt @ <sha>; do not edit in place — re-vendor instead.`).
3. Create `src/hssm/addm/likelihoods/jax/__init__.py` exposing only what HSSM needs downstream:
   ```python
   from .batch import (
       compute_addm_loglikelihoods,          # primary batched kernel
       compute_addm_loglikelihoods_from_mu,  # core wrapper for custom attention processes (see below)
       make_addm_nll_function,               # parity/reference only
   )
   from .multi_stage import compute_addm_logfptd  # parity tests only
   from .addm_helpers import _build_addm_mu_array_data
   from .utils import set_jax_precision, get_jax_dtype
   ```
4. **Expose one new public wrapper** in the vendored `batch.py`:
   `compute_addm_loglikelihoods_from_mu(rt, choice, mu_array, sacc_array, d, sigma, a, b, x0, *, ...)`,
   a thin public alias for the existing private core `_compute_addm_loglikelihoods_batchscan_core`.
   This is the only HSSM-authored edit inside the vendored tree; it gives the **custom attention
   process** path (Commit 2) a stable entry that accepts a pre-built drift array, while the
   default path keeps calling `compute_addm_loglikelihoods` (which builds the drift array
   internally). Record it in the header comment so a re-vendor preserves it.


**Not vendored:** `efficient_fpt/cython/`, `efficient_fpt/numpy/`, `efficient_fpt/models.py`,
the NumPy `addm_helpers.py`, the simulator. None are on the inference path.

No module outside `src/hssm/addm/` imports from `hssm.addm.likelihoods.jax` directly — keeping
the surface narrow makes a future re-vendor a single-directory change.

**Tests — `tests/addm/test_addm_vendored_jax.py`:**
1. `test_imports` — every symbol in the `jax/__init__.py` import block resolves.
2. `test_batched_runs_on_tiny_fixture` — `compute_addm_loglikelihoods` on a 5-trial hand-built
   batch returns shape `(5,)`, finite, dtype `== get_jax_dtype()`.
3. `test_batched_matches_per_trial` — batched output equals `compute_addm_logfptd` evaluated
   per trial (within `1e-6`) on the same fixture — confirms the retargeted imports didn't break
   the kernel.
4. `test_from_mu_matches_default` — `compute_addm_loglikelihoods_from_mu` with
   `mu_array = _build_addm_mu_array_data(...)` equals `compute_addm_loglikelihoods` on the same
   inputs (within `1e-6`) — confirms the new core wrapper is wired correctly.


### Commit 2 — `feat(addm): submodule scaffolding, attention process, likelihood Op builder`

Create the aDDM submodule mirroring [`src/hssm/rl/`](src/hssm/rl/):

```
src/hssm/addm/
  __init__.py               # exports aDDM, aDDMConfig (populated across commits 2–4)
  attention_process.py      # pluggable drift/fixation models (registry)
  config.py                 # aDDMConfig (Commit 3)
  addm.py                   # class aDDM(HSSMBase) (Commit 4)
  likelihoods/
    __init__.py
    builder.py              # make_addm_logp_func / make_addm_logp_op
    jax/                    # vendored (Commit 1)
```

**`attention_process.py` — the "learning process" analog.** A name→callable registry, mirroring
how `RLSSMConfig.learning_process` accepts a string or a mapping. The default
`standard_alternating` *is* the vendored drift logic:

```python
def standard_alternating(eta, kappa, r1, r2, flag, d, max_d):
    # Delegates to the vendored builder so the default attention process and the
    # default kernel agree by construction.
    return _build_addm_mu_array_data(eta, kappa, r1, r2, flag, d, max_d)

ATTENTION_PROCESSES = {"standard_alternating": standard_alternating}
```

Future variants (bias, drift offsets, non-alternating gaze) register by name. The registry is
the seam that keeps the class plan's extensibility story alive on the new batched API.

**`likelihoods/builder.py`**, mirroring [rl/likelihoods/builder.py](src/hssm/rl/likelihoods/builder.py)
(`make_rl_logp_func`, `make_rl_logp_op`):

1. **`make_addm_logp_func(attention_process="standard_alternating")`** → callable `logp(data, *args)`:
   - `data[:, 0]` = rt, `data[:, 1]` = response.
   - `args` are the sampled params in `list_params` order: `eta, kappa, a, b, x0`.
   - extra fields `r1, r2, flag, sacc_array, d, sigma` are appended to `args` by HSSM's extra-fields
     machinery (same as RLSSM; see `DataValidatorMixin._update_extra_fields` in
     [data_validator.py](src/hssm/data_validator.py)).
   - **Default attention process:** call `compute_addm_loglikelihoods(rt, response, eta, kappa,
     sigma, a, b, x0, r1, r2, flag, sacc_array, d)` directly (it builds the drift array
     internally via the same `_build_addm_mu_array_data`).
   - **Custom attention process:** call `mu = attention_process(eta, kappa, r1, r2, flag, d,
     sacc_array.shape[1])`, then `compute_addm_loglikelihoods_from_mu(rt, response, mu,
     sacc_array, d, sigma, a, b, x0)`.
2. **`make_addm_logp_op(attention_process, list_params, extra_fields)`** — wraps the JAX `logp`
   as a PyTensor `Op` with VJP, using the same construction as `make_rl_logp_op`. Gives NUTS
   gradients for free.

**Reuses:** `make_likelihood_callable` and `apply_param_bounds_to_loglik` from
[distribution_utils/dist.py](src/hssm/distribution_utils/dist.py). The Op is
**shape-polymorphic** (the kernel is a `lax.scan` over trials; JAX specializes on
`n_trials`/`max_d` at jit time), so it is built once per model instance and reused.

**Import contract:** `builder.py` imports the vendored likelihood as
`from .jax import compute_addm_loglikelihoods, compute_addm_loglikelihoods_from_mu, _build_addm_mu_array_data`.
Nothing else imports the vendored subpackage.

**Tests — `tests/addm/test_addm_builder.py`** (patterned after `tests/test_rl_builder_output_shape.py`):
1. `test_logp_func_output_shape` — `make_addm_logp_func()` output is `(n_trials,)`, dtype
   `get_jax_dtype()`.
2. `test_logp_op_matches_func` — the `Op` from `make_addm_logp_op()` returns the same values as
   the raw `logp` (within `1e-6`).
3. `test_logp_matches_kernel` — both match a direct `compute_addm_loglikelihoods` call on a
   10-trial fixture (within `1e-6`).
4. `test_gradients_finite` — `jax.grad` w.r.t. each of `eta, kappa, sigma, a, b, x0` is finite.
5. `test_custom_attention_process` — register a trivial custom process (e.g. scale drift by a
   constant) and assert it routes through `compute_addm_loglikelihoods_from_mu` and differs from
   the default in the expected direction.

### Commit 3 — `feat(addm): aDDMConfig dataclass`

Create **`src/hssm/addm/config.py`** with `aDDMConfig(BaseModelConfig)`, mirroring
[`RLSSMConfig`](src/hssm/rl/config.py#L25) (note: RLSSMConfig lives in `rl/config.py`, **not**
`config.py` — the class plan's "add a dataclass to config.py:457" is stale; we follow the
current per-submodule layout).

```python
@dataclass
class aDDMConfig(BaseModelConfig):
    """Config for the attentional DDM (subclass formulation)."""
    model_name: str = "addm"
    list_params: list[str] = field(
        default_factory=lambda: ["eta", "kappa", "a", "b", "x0"]
    )
    params_default: list[float] = field(
        default_factory=lambda: [0.3, 1.0, 1.0, 2.0, 0.0, 0.0]
    )
    response: list[str] = field(default_factory=lambda: ["rt", "response"])
    choices: tuple[int, ...] = (-1, 1)
    extra_fields: list[str] | None = field(
        default_factory=lambda: ["r1", "r2", "flag", "sacc_array", "d", "sigma"]
    )
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "eta":   (0.0, 1.0),
            "kappa": (0.0, 5.0),
            "a":     (0.1, 3.0),
            "b":     (0.0, 1.0),
            "x0":    (-1.0, 1.0),
        }
    )
    loglik_kind: str = "approx_differentiable"
    attention_process: str | Callable = "standard_alternating"
    # Filled in by aDDM.__init__ via dataclasses.replace, exactly like RLSSM:
    loglik: LogLik | None = None
    backend: str | None = None
    description: str | None = "Attentional Drift Diffusion Model"

    def validate(self) -> None:
        # attention_process resolves in the registry (or is callable); list_params and
        # bounds keys agree; extra_fields present. Mirrors RLSSMConfig.validate().
        ...

    @classmethod
    def get_defaults(cls) -> "aDDMConfig":
        return cls()

    @classmethod
    def from_addm_dict(cls, d: dict[str, Any]) -> "aDDMConfig":
        # Parallel to RLSSMConfig.from_rlssm_dict for dict-driven construction.
        ...
```

**Key decisions:**
- `extra_fields` defaults to the five aDDM covariates the kernel consumes (ordered to match the
  kernel's positional slots). They are **not** sampled parameters.
- `attention_process` is the pluggable hook (default `"standard_alternating"`), the analog of
  `RLSSMConfig.learning_process`.
- `loglik`/`backend` are left `None` here and **injected in `aDDM.__init__`** via
  `dataclasses.replace`, exactly as RLSSM does — keeping the caller's config object unmodified.
- `aDDM` is **not** added to `SupportedModels`; like `RLSSMConfig`, this config is constructed
  directly and never resolved through `get_default_model_config`.

**Tests — `tests/addm/test_addm_config.py`** (patterned after `tests/test_rlssm_config.py`):
1. `test_defaults` — `aDDMConfig()` has the expected `list_params`, `extra_fields`, `bounds`,
   `attention_process`, `model_name`.
2. `test_validate_ok` — `aDDMConfig().validate()` passes.
3. `test_validate_rejects_unknown_attention_process` — a bogus `attention_process` string raises.
4. `test_validate_rejects_bounds_param_mismatch` — `bounds` keys not matching `list_params` raises.
5. `test_from_addm_dict_roundtrip` — `from_addm_dict({...})` reproduces an equivalent config.

### Commit 4 — `feat(addm): aDDM(HSSMBase) subclass + data validation + top-level export`

The central commit: makes `hssm.aDDM(data=..., model_config=aDDMConfig(...))` construct and sample.

**File: `src/hssm/addm/addm.py`** — `class aDDM(HSSMBase)`, mirroring
[`RLSSM.__init__`](src/hssm/rl/rlssm.py#L110) but **without** the panel-reshape machinery:

```python
class aDDM(HSSMBase):
    def __init__(
        self,
        data: pd.DataFrame,
        model_config: aDDMConfig | None = None,   # defaults to aDDMConfig.get_defaults()
        include: list | None = None,
        p_outlier=0.05,
        lapse=bmb.Prior("Uniform", lower=0.0, upper=20.0),
        prior_settings="safe",
        missing_data: bool | float = False,
        deadline: bool | str = False,
        **kwargs,
    ) -> None:
        self._init_args = self._store_init_args(locals(), kwargs)
        model_config = model_config or aDDMConfig.get_defaults()
        model_config.validate()

        # aDDM-specific shape validation (sacc_array is 2-D inside a DataFrame column).
        # Lives on the subclass — no model-name-gated hook in the generic validator.
        self._validate_addm_columns(data, model_config)

        # Resolve the attention process to a concrete callable now, so the Op closure
        # captures it.
        attention_process = resolve_attention_process(model_config.attention_process)

        # Build the differentiable Op (fresh list() copies so later HSSMBase mutation of
        # list_params — appending "p_outlier" — is invisible to the Op's arg-length checks,
        # exactly as RLSSM documents).
        loglik_op = make_addm_logp_op(
            attention_process=attention_process,
            list_params=list(model_config.list_params),
            extra_fields=list(model_config.extra_fields or []),
        )
        model_config = replace(model_config, loglik=loglik_op, backend="jax")

        super().__init__(
            data=data, model_config=model_config, include=include,
            p_outlier=p_outlier, lapse=lapse, prior_settings=prior_settings,
            missing_data=missing_data, deadline=deadline, **kwargs,
        )

    def _make_model_distribution(self) -> type[pm.Distribution]:
        # Same shape as RLSSM._make_model_distribution: use the pre-built Op directly via
        # make_distribution, bypassing make_likelihood_callable. aDDM params are NOT
        # inherently trialwise (unlike RLSSM): params_is_trialwise is driven by which
        # params the user put under a regression, which HSSMBase already tracks.
        ...

    @staticmethod
    def _validate_addm_columns(data, model_config) -> None:
        # r1, r2, d, flag: 1-D numeric, length n_trials.
        # sacc_array: 2-D (n_trials, max_d), or a column of variable-length lists padded
        #   inline via the vendored padding helper.
        # d[i] <= sacc_array.shape[1]; flag in {0, 1}.
        ...
```

> **Note — unlike RLSSM, aDDM permits `missing_data`/`deadline`.** There is no within-participant
> ordering to corrupt, so these are forwarded to the base class rather than rejected. (If a
> concrete obstacle surfaces, gate them then; the default is to allow.)

**File: `src/hssm/addm/__init__.py`** — `from .addm import aDDM; from .config import aDDMConfig`.

**File: `src/hssm/__init__.py`** — export `aDDM` and `aDDMConfig` at the top level, next to
`RLSSM`/`RLSSMConfig`, so `hssm.aDDM(...)` is the entry point.

**No generic-module edits.** `_types.py`, `defaults.py`, `data_validator.py`, `config.py`,
`hssm.py`, `base.py` are **untouched** — aDDM validation lives on the subclass, and the Op
supersedes the `loglik` dispatch. (This is strictly cleaner than the old class plan, which
gated a hook inside `data_validator.py`.)

**Tests — `tests/addm/test_addm_subclass.py`:**
1. `test_construct_default_config` — `hssm.aDDM(data=tiny_df)` constructs (uses
   `aDDMConfig.get_defaults()`).
2. `test_construct_explicit_config` — `hssm.aDDM(data=tiny_df, model_config=aDDMConfig(...))`
   constructs.
3. `test_is_hssmbase_subclass` — `issubclass(hssm.aDDM, hssm.base.HSSMBase)` and the instance
   exposes the standard HSSMBase API (`.graph`, `.sample`, `.bounds`).
4. `test_loglik_op_injected` — after construction, `model.model_config.loglik` is a PyTensor `Op`
   and `model.model_config.backend == "jax"`; the caller's original `aDDMConfig` is unmodified.
5. `test_bad_sacc_shape_raises` — a `sacc_array` with `d[i] > max_d`, a non-binary `flag`, or a
   length-mismatched `r1` raises a clear error from `_validate_addm_columns`.
6. `test_smoke_sample` — 200-trial synthetic dataset; `hssm.aDDM(data=trials).sample(draws=5,
   tune=5)` returns an `InferenceData`.
7. `test_hierarchical_regression_builds` — `include=[{"name": "eta", "formula": "eta ~ 1 + (1|participant)"}]`
   builds without error (showcases what the subclass buys over raw efficient-fpt).

**Synthetic data:** `efficient_fpt.aDDModel(...).generate_experiment(...)` (dev-only import),
materialized as a pandas DataFrame with columns `rt, response, r1, r2, flag, sacc_array, d`.

### Commit 5 — `feat(addm): non-decision time t as a sampled pre-decision motor delay`

Adds `t` to `list_params` and shifts both `rt` and `sacc_array` by `t` before the JAX kernel
runs. **Interpretation:** `t` is a pre-decision motor/encoding delay — during `[0, t]` neither
the decision diffusion nor the recorded fixations contribute to drift evolution.

**Why this is more than "subtract `t` from `rt`":** `sacc_array` is in *trial-time* (with
`sacc_array[:, 0] = 0`) while the decision-only likelihood expects *decision-time*. A naive
`rt - t` without touching `sacc_array` would mis-align fixation-stage onsets relative to the
decision process:

| Quantity | Trial-time (input) | Decision-time (to kernel) |
|---|---|---|
| Response time | `rt` | `rt - t` |
| Stage 0 onset | `sacc_array[:, 0] = 0` | `0` (anchored to decision start) |
| Stage `i ≥ 1` onset | `sacc_array[:, i]` | `sacc_array[:, i] - t` |
| Stage count `d`, flag `flag` | `d`, `flag` | unchanged (see constraint) |

**Critical constraint — `t` lives entirely inside the first fixation:** `t < sacc_array[:, 1]`
for every trial. Consequences:
- The first stage's effective decision-time duration shrinks from `sacc_array[:, 1]` to
  `sacc_array[:, 1] - t` (truncated head); all later stage durations are preserved.
- No stage is dropped, so the first-fixation `flag` never flips.
- The transformation is smooth in `t`, so NUTS gets clean gradients everywhere except the
  boundary `t = sacc_array[:, 1]`, which is bounded out of the support.

This matches the literature: motor/encoding NDT is ~150–400 ms while first fixations are
typically 500 ms+.

**Code change — `src/hssm/addm/likelihoods/builder.py`** (`make_addm_logp_func` gains `t`):

```python
def logp(data, eta, kappa, sigma, a, b, x0, t, r1, r2, flag, sacc_array, d):
    rt = data[:, 0]; response = data[:, 1]
    rt_shifted = rt - t
    sacc_shifted = sacc_array.at[:, 1:].add(-t)          # JAX out-of-place; index 0 stays 0
    first_sacc_end = sacc_array[:, 1]                     # max_d >= 2 guaranteed by validator
    valid = (rt_shifted > 0.0) & (t < first_sacc_end)
    ll = compute_addm_loglikelihoods(
        rt_shifted, response, eta, kappa, sigma, a, b, x0,
        r1, r2, flag, sacc_shifted, d,
    )
    return jnp.where(valid, ll, -jnp.inf)
```

**Caveats baked in:**
1. **`t = 0` is exact** — `rt_shifted = rt`, `sacc_shifted = sacc_array`; the kernel sees exactly
   the no-NDT arrays. This commit is strictly a superset of Commits 1–4.
2. **`-inf` masking, not error** — a proposed `t` that violates the constraint on a trial returns
   `-inf` for that trial; NUTS rejects the proposal cleanly. Raising would crash the sampler.
3. **`jnp.where`, not `lax.cond` on `t`** — keeps the trace shape-stable (no recompilation); the
   masked-out invalid trials are computed-then-discarded.
4. **`max_d >= 2`** — the validator is extended to reject `d < 2`, so `sacc_array[:, 1]` is always
   meaningful. Single-fixation trials are a degenerate case the pre-decision interpretation
   doesn't support; reject them (see Open Question 3) **This I'm not entirely sure of -- are we supposed to reject single-fixation trials here? - Andrew**.
5. **Gradient through `.at[:, 1:].add(-t)`** — JAX's scatter-add propagates `∂/∂t` correctly.
6. **Precision** — `rt - t` and early `sacc[:, i] - t` can be small; run this commit's tests under
   `set_jax_precision("x64")` and document it.

**Config change — `src/hssm/addm/config.py`:** append `"t"` to `list_params` and add the bound
`"t": (0.0, 0.4)` (conservative; users with short first fixations override via
`include=[{"name": "t", "prior": ..., "bounds": ...}]`). `params_default` gains a small `t`
(e.g. `0.0`).

**Validator change — `aDDM._validate_addm_columns`:**
- Assert `sacc_array.shape[1] >= 2` and every trial `d >= 2`.
- Warn (don't fail) if the model's default `t` upper bound exceeds `min(sacc_array[:, 1])` over
  the dataset — that range routinely violates the in-first-fixation constraint and wastes
  gradient evaluations. Suggest tightening (Open Question 2).

**Tests — `tests/addm/test_addm_ndt.py`** (run under `x64`):
1. `test_ndt_zero_matches_no_ndt` — `t = 0` matches the Commit-4 no-NDT logp bit-for-bit.
2. `test_ndt_shifts_logp_consistently` — for `t > 0`, the logp equals a manual
   `compute_addm_loglikelihoods(rt - t, ..., sacc_shifted, ...)`.
3. `test_ndt_invalid_t_returns_neginf` — `t > sacc_array[i, 1]` for one trial → that trial's logp
   is `-inf`, others finite.
4. `test_ndt_gradient_finite` — `jax.grad` w.r.t. `t` is finite at an interior valid point.
5. `test_validator_rejects_single_stage` — a dataset with any `d < 2` raises.
6. `test_ndt_smoke_sample` — `hssm.aDDM(data=trials).sample(draws=5, tune=5)` with `t` in
   `list_params` returns an `InferenceData`.

> **Why a separate commit.** Landing `t` after the no-NDT path is proven isolates the extra
> complexity (sacc-shift, validity masking) from the core-correctness story. If the NDT
> interpretation later needs elaboration for some dataset, Commits 1–4 remain a working baseline.

### Commit 6 — `feat(addm): user covariates + fixation continuation in the cssm.addm PPC simulator` *(ssm-simulators, cross-repo)*

**Two coupled problems, one commit.** Faithful posterior predictives require simulating under the
fitted model *on the dataset's own covariates*. Today `cssm.addm`
(`ssm-simulators/src/cssm/addm_models.pyx`) does the opposite on both counts:

1. **No covariate passthrough.** It randomizes every trial-level latent internally — `r1`, `r2`,
   `flag`, and the fixation durations — and exposes no input for them. There is no saccade-array
   argument, so it can only produce a *generic* aDDM predictive, never one conditioned on a
   specific trial's stimulus or observed gaze.
2. **Last-fixation freeze.** It pre-generates a fixed budget of `max_fixations` durations and looks
   up drift via `_piecewise_drift`, which **returns the last stage's drift for any `t` beyond the
   final saccade**. When a `(trial, sample)`'s decision outruns the available fixations, the
   particle is held at one fixed drift and it
   distorts the RT/choice tails. Under self-generation the default budget over-covers `max_t`, so
   it is *masked*; but with a **finite user-supplied saccade prefix** (human fixation onsets,
   which only extend to the observed RT) a slow draw runs past the prefix and the freeze is
   *guaranteed*.

These couple: once users can supply real (finite) saccade arrays, the freeze becomes the common
failure mode. So both land together. (efficient-fpt's own `_run_heterog_trial` has the identical
freeze and already accepts a `sacc_array_data` prefix; the target design unifies both —
efficient-fpt's "accepts a prefix" plus a continuation efficient-fpt lacks.)

**Design — optional covariates + a spliced fixation sequence.** Add optional per-trial covariate
inputs to `cssm.addm`; when `None`, fall back to current internal sampling (full backward
compatibility).

| Input (per trial) | When supplied | When `None` (default) |
|---|---|---|
| `r1`, `r2` | use as stimulus ratings | randomize `int∈[1,5]` |
| `flag` | use as first-item indicator | randomize `0/1` |
| `sacc_array` (padded `n×max_d`) + `d` | use as the **observed fixation prefix** | empty prefix → fully self-generated based off of subject-wise fixation data |

The fixation sequence used by the diffusion is always **prefix ++ continuation**:
- *Prefix* = supplied observed onsets (or empty).
- *Continuation* = `Gamma(gamma_shape, gamma_scale)` fits a Gamma distribution based off of subject-wise fixation data until cumulative time
  `≥ deadline_tmp = compute_deadline_tmp(max_t, deadline, t)`. Because the boundary collapses,
  `deadline_tmp` is finite (bounded by `a/b` and `max_t`), so coverage is guaranteed and
  `_piecewise_drift` can never reach "beyond the last saccade" within an active trial — **the
  freeze becomes structurally impossible in every regime.**
- Continuation preserves the alternation: stage `i ≥ d_prefix` takes the item opposite stage
  `i−1`, derived from `flag` and parity, so the spliced drift array stays consistent across the
  prefix→continuation seam.

Sketch (`addm_models.pyx`, replacing the fixed-budget block; `prefix` = supplied onsets or `[0.0]`):
```python
sacc_list = list(prefix) if prefix is not None else [0.0]
cum = sacc_list[-1]
while cum < deadline_tmp:
    cum += rng.gamma(gamma_shape, gamma_scale) # ought to be a Gamma with parameters extrapolated per subject
    sacc_list.append(cum)
sacc_np = np.asarray(sacc_list, dtype=np.float64); d = len(sacc_np)
# mu_np built by alternating from `flag`, parity continued across the seam
```
`max_fixations` becomes a soft safety cap on *continuation* length (warn/raise under pathological
params), not the coverage mechanism. Per `(trial, sample)` the prefix is reused and the
continuation is freshly drawn (a fresh gaze realization beyond the observed data).

**Coordinate consistency with NDT (Commit 5).** Supplied onsets are in **trial-time**; the
simulator runs in **decision-time**. Before simulating, shift the prefix by the (simulated/
sampled) `t` exactly as Commit 5 shifts the likelihood inputs (`sacc_array[:, 1:] − t`, stage 0
anchored at 0), drop any prefix onset `≤ t`, and add `t` back to the simulated decision time at
the end (`cssm.addm` already does `rts = t_particle + t + smooth_u`). Continuation is generated in
decision-time. This keeps simulator and likelihood in the same frame.

**HSSM-side plumbing (this repo) — on the subclass.** Override the PPC/simulate path **on
`aDDM`** so it forwards the dataset's `extra_fields` (`r1, r2, flag, sacc_array, d`) into
`cssm.addm`. Concretely, add `aDDM._simulate_kwargs_from_data()` (or override the relevant
posterior-predictive hook) that pulls the covariate columns and passes them through; document
that omitting them yields the generic self-generated predictive. No likelihood change. Because
aDDM is a subclass, this plumbing is self-contained — no model-name branching in generic code.

**Emit the generated fixation sequence.** Extend the `full` return metadata with the per-
`(trial, sample)` spliced `sacc_np`/`d`/`flag`, so PPC can check predicted *fixation counts* and
gaze statistics, not just `rt`/`choice`. (Minimal-return path unchanged.)


**Tests:**
- *ssm-simulators (`cssm.addm` suite):*
  1. `test_covariates_respected` — supplied `r1`/`r2`/`flag` are used verbatim (drifts match
     `kappa·(r1 − eta·r2)` etc.), not randomized.
  2. `test_supplied_prefix_used` — supplied prefix + fast decision → simulated switch-times match
     the prefix up to `d_prefix`.
  3. `test_continuation_past_prefix` — short prefix + slow decision → sequence extends past
     `d_prefix`, alternates across the seam, reaches `deadline_tmp`..
  4. `test_fixation_coverage_invariant` — `sacc_np[-1] ≥ deadline_tmp` for every `(trial, sample)` (**not sure if needed - Andrew**).
  5. `test_ndt_coordinate_consistency` — a trial-time prefix shifted by `t` yields the same
     decision-time sequence as a manual Commit-5 shift (**not sure if needed - Andrew**).
  6. `test_fixation_metadata_emitted` — `return_option='full'` includes per-sample spliced
     sequences with sane counts (**not sure if needed - Andrew**).
- *HSSM (`tests/addm/test_addm_ppc.py`):*
  7. `test_ppc_forwards_covariates` — `aDDM(...).sample_posterior_predictive(...)` (or the
     simulate path) passes the dataset's `r1/r2/flag/sacc_array/d` to `cssm.addm`; predictive RTs
     are conditioned on them (differ from the covariate-free predictive).


> **Optional split.** Can land as **6a** (covariate passthrough: inputs + plumbing + tests 1–2,
> 4, 8) and **6b** (continuation + coverage guarantee: tests 3, 5–7, 9). 6b is only fully
> exercised once 6a allows finite user prefixes, so keep them adjacent.

### Commit 7 — `test+docs(addm): parameter recovery, tutorial notebook, docs`

- **`tests/scripts/addm_recovery.py`** (off-CI) — 1000-trial recovery with known
  `(eta, kappa, a, b, x0, t)`; fit via `hssm.aDDM(data=...)`; confirm posterior means
  within ~2σ of ground truth. Reuse the recovery setup from
  [efficient-fpt examples/example8_empirical](data/azhang/efficient-fpt/examples/example8_empirical).
- **`docs/tutorials/addm_tutorial.ipynb`** — mirror
  [docs/tutorials/rlssm_tutorial.ipynb](docs/tutorials/rlssm_tutorial.ipynb):
  1. Generate a small aDDM dataset two ways: `efficient_fpt.aDDModel(...).generate_experiment(...)`
     and/or the `cssm.addm` simulator via HSSM's simulate path.
  2. Build the model with **`hssm.aDDM(data=..., model_config=aDDMConfig(...), include=[...])`** —
     the canonical subclass path.
  3. Hierarchical regression on `eta` by participant, to showcase what HSSM buys over raw
     efficient-fpt.
  4. `model.sample()` and posterior/PPC plots via `arviz` (including predicted fixation counts
     from Commit 6's metadata).
- **`mkdocs.yml`** — add the tutorial to nav. **`README.md`** — add `aDDM` to the model list.

**Tests for this commit:**
1. `test_recovery_within_2sigma` *(off-CI marker)* — the recovery script's assertions pass.
2. `test_tutorial_executes` *(CI-optional)* — `jupyter nbconvert --execute
   docs/tutorials/addm_tutorial.ipynb` finishes without error (or a fast-mode cut-down variant in
   CI).
3. `test_docs_build` — `mkdocs build` succeeds with the new nav entry.

### Commit 8 — `chore(addm): cleanup`

- Delete the stale `addm-andrew-dev/` scratch folder (note its trailing-space variant on some
  filesystems is a foot-gun) once the new module works.
- Bump the ssm-simulators floor in `pyproject.toml` to the first release containing the Commit-6
  `cssm.addm` covariate/continuation fix.
- Refresh the upstream commit hash in `src/hssm/addm/likelihoods/jax/NOTICE` and the per-file
  vendor headers.

**Tests for this commit:**
1. `test_no_stale_scratch_dir` — the scratch folder is gone (or repo-lint passes).
2. `test_ssm_simulators_floor` — installed ssm-simulators version `>=` the pinned floor.

---

## Files created (cumulative)

**HSSM-original code** (`src/hssm/addm/`):
- `src/hssm/addm/__init__.py` — exports `aDDM`, `aDDMConfig`.
- `src/hssm/addm/config.py` — `aDDMConfig(BaseModelConfig)`.
- `src/hssm/addm/addm.py` — `class aDDM(HSSMBase)`.
- `src/hssm/addm/attention_process.py` — attention-process registry + `standard_alternating`.
- `src/hssm/addm/likelihoods/__init__.py`
- `src/hssm/addm/likelihoods/builder.py` — `make_addm_logp_func`, `make_addm_logp_op`.

**Vendored from efficient-fpt `efficient_fpt/jax/`** (own subpackage):
- `src/hssm/addm/likelihoods/jax/__init__.py`
- `src/hssm/addm/likelihoods/jax/batch.py`
- `src/hssm/addm/likelihoods/jax/multi_stage.py`
- `src/hssm/addm/likelihoods/jax/single_stage.py`
- `src/hssm/addm/likelihoods/jax/addm_helpers.py`
- `src/hssm/addm/likelihoods/jax/utils.py`
- `src/hssm/addm/likelihoods/jax/_defaults.py`
- `src/hssm/addm/likelihoods/jax/NOTICE`

**Tests & docs:**
- `tests/addm/test_addm_vendored_jax.py` *(Commit 1)*
- `tests/addm/test_addm_builder.py` *(Commit 2)*
- `tests/addm/test_addm_config.py` *(Commit 3)*
- `tests/addm/test_addm_subclass.py` *(Commit 4)*
- `tests/addm/test_addm_ndt.py` *(Commit 5)*
- `tests/addm/test_addm_ppc.py` *(Commit 6)*
- `tests/scripts/addm_recovery.py` *(Commit 7, off-CI)*
- `docs/tutorials/addm_tutorial.ipynb` *(Commit 7)*

## Files modified

- `src/hssm/__init__.py` — export `aDDM`, `aDDMConfig` (peer of `RLSSM`, `RLSSMConfig`).
- `pyproject.toml` — bump ssm-simulators floor (Commit 8).
- `README.md`, `mkdocs.yml` — mention the model and tutorial (Commit 7).
- *(cross-repo)* `ssm-simulators/src/cssm/addm_models.pyx` — covariate passthrough + fixation
  continuation (Commit 6).

**Not modified — and why (the subclass approach keeps generic HSSM code untouched):**
- `src/hssm/_types.py` — aDDM is **not** a `SupportedModels` entry; like RLSSM, it is a subclass,
  not a registered model.
- `src/hssm/defaults.py` — no lazy `<name>_config` resolution; `aDDMConfig` is constructed directly.
- `src/hssm/data_validator.py` — aDDM column validation lives on the `aDDM` subclass, not as a
  model-name-gated hook in the generic validator.
- `src/hssm/config.py` — `aDDMConfig` lives in `src/hssm/addm/config.py` (mirroring
  `RLSSMConfig` in `rl/config.py`), not in `config.py`.
- `src/hssm/hssm.py` / `src/hssm/base.py` — no edits; `aDDM` subclasses `HSSMBase` and overrides
  only `_make_model_distribution`.

## Key functions / utilities reused (no re-implementation)

| Purpose | Location |
|---|---|
| **Batched JAX FPT log-likelihood** | `hssm.addm.likelihoods.jax.compute_addm_loglikelihoods` *(vendored from `efficient_fpt.jax.batch`)* |
| Core wrapper for custom attention processes | `hssm.addm.likelihoods.jax.compute_addm_loglikelihoods_from_mu` *(vendored, public alias of the batchscan core)* |
| Single-trial JAX (parity tests) | `hssm.addm.likelihoods.jax.compute_addm_logfptd` *(vendored from `efficient_fpt.jax.multi_stage`)* |
| Padded drift-array constructor (default attention process) | `hssm.addm.likelihoods.jax._build_addm_mu_array_data` *(vendored from `efficient_fpt.jax.addm_helpers`)* |
| JAX precision control | `hssm.addm.likelihoods.jax.set_jax_precision` |
| **Subclass pattern** (`__init__` Op-build + `_make_model_distribution` override) | [`RLSSM`](src/hssm/rl/rlssm.py#L41) |
| **Config dataclass pattern** | [`RLSSMConfig`](src/hssm/rl/config.py#L25) on [`BaseModelConfig`](src/hssm/config.py#L38) |
| Likelihood `Op` builder pattern | [rl/likelihoods/builder.py](src/hssm/rl/likelihoods/builder.py) |
| `LogLik` typedef (accepts `Op`) | [`hssm._types.LogLik`](src/hssm/_types.py#L11) |
| Distribution assembly from a pre-built Op | `hssm.distribution_utils.make_distribution` |
| Extra-fields propagation into logp | [`data_validator.DataValidatorMixin._update_extra_fields`](src/hssm/data_validator.py) |
| Param bound enforcement | [`distribution_utils.dist.apply_param_bounds_to_loglik`](src/hssm/distribution_utils/dist.py) |
| **aDDM simulator (prior/posterior predictives)** | `cssm.addm` via ssm-simulators |
| Synthetic dataset generation (tests) | `efficient_fpt.aDDModel.generate_experiment` *(dev-only import; not vendored)* |

---

## Verification

End-to-end checks, in order:

1. **Vendored kernel** — `pytest tests/addm/test_addm_vendored_jax.py` passes (Commit 1).
2. **Builder/Op** — `pytest tests/addm/test_addm_builder.py` passes: shapes, Op↔func↔kernel
   parity to `1e-6`, finite gradients, custom attention process (Commit 2).
3. **Config** — `pytest tests/addm/test_addm_config.py` passes: defaults, `validate`,
   `from_addm_dict` round-trip (Commit 3).
4. **Subclass construct + smoke** — `hssm.aDDM(data=tiny_df)` constructs; `.sample(draws=5,
   tune=5)` returns an `InferenceData`; `issubclass(hssm.aDDM, HSSMBase)` (Commit 4).
5. **NDT correctness** — `t = 0` matches the no-NDT path bit-for-bit; `t > 0` matches a manual
   `rt`/`sacc_array` shift; constraint-violating `t` returns `-inf` per offending trial; `d < 2`
   rejected (Commit 5, run under `x64`).
6. **Simulator path** — `aDDM` PPC forwards `r1/r2/flag/sacc_array/d` into `cssm.addm`;
   covariate-conditioned predictives differ from the covariate-free predictive; no
   fixation-exhaustion tail spike; `sacc_np[-1] ≥ deadline_tmp` per sample; omitting covariates
   still yields the generic self-generated predictive (Commit 6).
7. **Parameter recovery** (off-CI) — `tests/scripts/addm_recovery.py` confirms posterior means
   within ~2σ of ground truth on 1000 trials, including `t` (Commit 7).
8. **Tutorial runs clean** — `jupyter nbconvert --execute docs/tutorials/addm_tutorial.ipynb`
   finishes without errors (Commit 7).
9. **Docs build** — `mkdocs build` succeeds with the new tutorial in nav (Commit 7).
10. **Cleanup** — stale scratch dir gone; ssm-simulators floor bumped; NOTICE hash refreshed
    (Commit 8).

## Open questions

1. **Default `model_config`.** We make `aDDM(data=df)` work by defaulting to
   `aDDMConfig.get_defaults()`. RLSSM requires an explicit config; confirm the more permissive
   default is desired (it is more user-friendly and we keep the explicit path too).
2. **`t` upper bound.** `t ∈ (0.0, 0.4)` may be too loose for short first fixations. If the
   Commit-5 warning fires often, auto-tighten to `(0.0, 0.9 · min(sacc_array[:, 1]))` per dataset,
   or expose a helper that suggests a bound?
3. **Single-stage trials (`d == 1`) What do we do in the case of single-stage trials?.** Rejected by the Commit-5 validator. If real datasets carry
   a non-trivial fraction of single-fixation trials, add a fallback (pure-DDM likelihood for those
   trials)?
4. **`missing_data`/`deadline`.** Allowed for aDDM (no within-participant ordering to corrupt,
   unlike RLSSM). Confirm no hidden coupling in the simulate/PPC path before advertising support. Not entirely sure what the purpose would be for 'missing_data' and 'deadline'
5. **Convenience vs explicitness.** Should `hssm.HSSM` gain any awareness of aDDM, or is
   `hssm.aDDM` the sole entry point? This plan keeps it sole (mirrors `hssm.RLSSM`). Down the line, may want to incorporate ADDM as a model under the main HSSM class. 
