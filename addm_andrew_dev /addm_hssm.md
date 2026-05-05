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

The intended outcome is a working `aDDM(...)` class (and matching `aDDMConfig`) inside HSSM that (a) validates aDDM-specific trial data, (b) composes the vendored JAX FPT likelihood with sampled parameters `{eta, kappa, sigma, a, b, x0, t}` (non-decision time optional), (c) exposes the standard HSSM hierarchical regression and sampling machinery, and (d) ships with a tutorial notebook and unit tests.

## Design choice: subclass pattern (matching the new RLSSM architecture)

> **Architecture update (post-rebase, commit `b4228d1b` — "HSSM base + RLSSM classes" #893).** The HSSM base was refactored: `HSSMBase` is now an abstract base ([src/hssm/base.py](data/azhang/HSSM/src/hssm/base.py)) and both `HSSM` ([src/hssm/hssm.py](data/azhang/HSSM/src/hssm/hssm.py)) and `RLSSM` ([src/hssm/rl/rlssm.py](data/azhang/HSSM/src/hssm/rl/rlssm.py)) are concrete subclasses. `RLSSMConfig` was moved out of `hssm.config` into [src/hssm/rl/config.py](data/azhang/HSSM/src/hssm/rl/config.py) and **no longer exposes a `to_config()` method** — `HSSMBase` accepts any `BaseModelConfig` directly, and the family-specific subclass is responsible for building its own likelihood `Op` and stamping it onto the config via `dataclasses.replace(...)` before calling `super().__init__()`. The new `RLSSMConfig` also adds an `ssm_logp_func` field (an `@annotate_function`-decorated JAX function) and renamed `learning_process_loglik_kind` → `learning_process_kind`.

Given this new architecture, the plan creates **both**:

1. An `aDDM(HSSMBase)` concrete subclass — a peer of `HSSM` and `RLSSM`, exported from `hssm/__init__.py` as `hssm.aDDM`.
2. An `aDDMConfig(BaseModelConfig)` dataclass living in `src/hssm/addm/config.py`, peer of `RLSSMConfig` in `src/hssm/rl/config.py`.

The `aDDM` subclass handles (i) validating aDDM-specific data shape, (ii) building the differentiable PyTensor `Op` from the vendored JAX likelihood, (iii) stamping that `Op` onto the config (via `replace(loglik=op, backend="jax")`), and (iv) overriding `_make_model_distribution` to bypass the standard `loglik_kind` dispatching — exactly mirroring `RLSSM.__init__` and `RLSSM._make_model_distribution`.

(There is **no longer** a `to_config()` round-trip; that method existed only in the pre-rebase architecture and has been removed from `RLSSMConfig` as well.)

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

Create a new package under `src/hssm/addm/`, mirroring the **post-rebase** `src/hssm/rl/` layout (`config.py` and `rlssm.py` at the package root, plus a `likelihoods/` subpackage):

```
src/hssm/addm/
  __init__.py             # exports aDDM, aDDMConfig
  config.py               # aDDMConfig — peer of hssm.rl.config.RLSSMConfig
  addm.py                 # aDDM(HSSMBase) — peer of hssm.rl.rlssm.RLSSM
  attention_process.py    # pluggable fixation/attention models (default: standard_alternating)
  utils.py                # validate_addm_panel(...) — peer of hssm.rl.utils.validate_balanced_panel
  likelihoods/
    __init__.py
    builder.py            # make_addm_logp_func / make_addm_logp_op
    addm_jax.py           # thin wrapper that imports from .jax and applies the attention process
    jax/                  # vendored from efficient_fpt_jax (Step 1)
      __init__.py
      multi_stage.py
      single_stage.py
      utils.py
```

**Rationale:** aDDM is conceptually a two-stage model (attention process → SSM likelihood) just like RLSSM (learning process → SSM likelihood). After the rebase, RLSSM is a real `HSSMBase` subclass with its own subpackage; the aDDM subpackage now mirrors that exactly — same files (`config.py`, `<model>.py`, `utils.py`, `likelihoods/builder.py`), same conventions. The vendored JAX code lives in its own `jax/` subdirectory so it stays clearly identifiable as upstream-derived, isolated from HSSM-original code in `builder.py` and `addm_jax.py`.

### Step 3 — Add `aDDMConfig` dataclass in `src/hssm/addm/config.py`

**Critical file:** [src/hssm/addm/config.py](data/azhang/HSSM/src/hssm/addm/config.py) (new file) — peer of [src/hssm/rl/config.py](data/azhang/HSSM/src/hssm/rl/config.py).

> **Note:** `aDDMConfig` is **not** added to `src/hssm/config.py`. After the rebase, family-specific configs live in their own subpackages (`hssm.rl.config.RLSSMConfig`, `hssm.addm.config.aDDMConfig`), with only `BaseModelConfig`, `Config`, and `ModelConfig` remaining in `hssm.config`.

```python
from dataclasses import dataclass, field
from typing import Any, Callable
from ..config import BaseModelConfig, DEFAULT_SSM_OBSERVED_DATA, DEFAULT_SSM_CHOICES

@dataclass
class aDDMConfig(BaseModelConfig):
    """Config for the attentional DDM."""

    # Required (kw_only) fields — pattern borrowed from RLSSMConfig
    params_default: list[float] = field(kw_only=True)
    attention_process: str | Callable | dict[str, Any] = field(
        kw_only=True, default="standard_alternating"
    )

    # aDDM-specific extra-field column names (defaultable)
    # These are *data* columns, not sampled parameters.
    extra_fields: list[str] | None = field(
        default_factory=lambda: ["r1", "r2", "sacc_array", "d", "flag"]
    )

    def __post_init__(self):
        if self.loglik_kind is None:
            self.loglik_kind = "approx_differentiable"

    @classmethod
    def from_defaults(cls, model_name, loglik_kind):
        raise NotImplementedError(
            "aDDMConfig does not support from_defaults(). "
            "Use the aDDM(...) constructor directly, or pass an aDDMConfig "
            "instance built explicitly."
        )

    def validate(self) -> None:
        # Mirror RLSSMConfig.validate: required fields, params_default vs
        # list_params length parity, every list_params entry has bounds, etc.
        ...

    def get_defaults(self, param):
        return None, self.bounds.get(param)
```

**Key design decisions (post-rebase):**

- **No `to_config()` method.** The new architecture has `HSSMBase` accept any `BaseModelConfig`; family-specific subclasses build the `loglik` `Op` themselves and stamp it onto the config via `dataclasses.replace(...)`. `RLSSMConfig` no longer has `to_config()` and neither will `aDDMConfig`.
- **No `from_defaults` registration.** Like `RLSSMConfig`, `aDDMConfig` raises `NotImplementedError` from `from_defaults`. Users construct it explicitly (or via a `from_addm_dict` classmethod, optional). Therefore **`aDDM` is *not* registered through the `default_model_config` / `register_model` pipeline** that `HSSM(model="ddm", ...)` uses — instead, users instantiate `hssm.aDDM(...)` directly.
- `extra_fields` defaults to the five aDDM-specific columns the JAX likelihood needs; these flow through the existing extra-fields machinery (data validator → `Op` `*args`) the same way they do for RLSSM.
- `attention_process` is a pluggable hook (default `"standard_alternating"`) that maps `(r1, r2, flag, eta, kappa) → mu_array_padded` per trial. This mirrors `RLSSMConfig.learning_process` semantically (declarative documentation; the actual callable is resolved by the builder).
- `list_params` covers the sampled parameters: `["eta", "kappa", "sigma", "a", "b", "x0"]`. Non-decision time `t` is deliberately omitted initially.
- `bounds` is required (every `list_params` entry must have an entry), matching `RLSSMConfig.validate`'s post-rebase strictness.

**Reuses:** `BaseModelConfig` ([src/hssm/config.py:48](data/azhang/HSSM/src/hssm/config.py#L48)), defaults `DEFAULT_SSM_OBSERVED_DATA` / `DEFAULT_SSM_CHOICES` ([src/hssm/config.py:24-26](data/azhang/HSSM/src/hssm/config.py#L24-L26)).

### Step 3b — Add `aDDM` subclass in `src/hssm/addm/addm.py`

**Critical file:** [src/hssm/addm/addm.py](data/azhang/HSSM/src/hssm/addm/addm.py) (new file) — peer of [src/hssm/rl/rlssm.py](data/azhang/HSSM/src/hssm/rl/rlssm.py).

This step is **new in the post-rebase plan.** Mirror `RLSSM` (264 lines):

```python
from dataclasses import replace
from typing import Any, Callable, Literal, cast
import bambi as bmb
import pandas as pd
import pymc as pm

from hssm.distribution_utils import make_distribution
from hssm.defaults import INITVAL_JITTER_SETTINGS
from ..base import HSSMBase
from .config import aDDMConfig
from .likelihoods.builder import make_addm_logp_op
from .utils import validate_addm_panel


class aDDM(HSSMBase):
    def __init__(
        self,
        data: pd.DataFrame,
        model_config: aDDMConfig,
        include: list | None = None,
        p_outlier: float | dict | bmb.Prior | None = 0.05,
        lapse: dict | bmb.Prior | None = bmb.Prior("Uniform", lower=0.0, upper=20.0),
        link_settings: Literal["log_logit"] | None = None,
        prior_settings: Literal["safe"] | None = "safe",
        extra_namespace: dict | None = None,
        missing_data: bool | float = False,
        deadline: bool | str = False,
        loglik_missing_data: Callable | None = None,
        process_initvals: bool = True,
        initval_jitter: float = INITVAL_JITTER_SETTINGS["jitter_epsilon"],
        **kwargs,
    ):
        self._init_args = self._store_init_args(locals(), kwargs)
        model_config.validate()

        # Same row-order argument as RLSSM: per-trial sacc_array shape and the
        # JAX vmap rely on strict 1:1 row→trial correspondence; reordering
        # missing/deadline rows would break the alignment.
        if missing_data is not False or deadline is not False:
            raise NotImplementedError(
                "aDDM does not support `missing_data` or `deadline` handling..."
            )

        validate_addm_panel(data, model_config.extra_fields)

        loglik_op = make_addm_logp_op(
            attention_process=model_config.attention_process,
            data_cols=list(model_config.response),
            list_params=list(model_config.list_params),
            extra_fields=list(model_config.extra_fields or []),
        )

        # Stamp the Op + backend onto a fresh config copy (do NOT mutate the
        # caller's config). Identical pattern to RLSSM.__init__.
        model_config = replace(model_config, loglik=loglik_op, backend="jax")

        super().__init__(
            data=data, model_config=model_config, include=include,
            p_outlier=p_outlier, lapse=lapse,
            link_settings=link_settings, prior_settings=prior_settings,
            extra_namespace=extra_namespace,
            missing_data=missing_data, deadline=deadline,
            loglik_missing_data=loglik_missing_data,
            process_initvals=process_initvals, initval_jitter=initval_jitter,
            **kwargs,
        )

    def _make_model_distribution(self) -> type[pm.Distribution]:
        # Mirror RLSSM._make_model_distribution: bypass loglik_kind dispatching
        # and feed the pre-built Op directly into make_distribution.
        ...
```

**Key design decisions:**

- **`aDDM` is a peer of `HSSM` and `RLSSM`**, all three inheriting from `HSSMBase`. It is exported from `hssm/__init__.py` as `hssm.aDDM`.
- **`_make_model_distribution` is overridden** (same as `RLSSM`) because the aDDM `Op` already encapsulates the attention process + per-trial vmap; the standard `make_likelihood_callable` dispatch on `loglik_kind` should be bypassed.
- **No `participant_col`-style panel reshape** — unlike RLSSM (which reshapes rows into `(n_participants, n_trials, ...)` because the RL learning rule is per-subject), aDDM's likelihood is per-trial. The vmap inside the JAX `Op` handles the trial dimension; participants flow through bambi/HSSM hierarchical regression as usual.
- **`missing_data` / `deadline` are rejected up front**, same as RLSSM, because rearranging rows would break the strict trial→`sacc_array`-row correspondence.

**Reuses:** `HSSMBase` ([src/hssm/base.py:92](data/azhang/HSSM/src/hssm/base.py#L92)), `make_distribution` ([src/hssm/distribution_utils](data/azhang/HSSM/src/hssm/distribution_utils)), `INITVAL_JITTER_SETTINGS` (defaults.py).

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

### Step 6 — Export `aDDM` and `aDDMConfig` from the top-level package

**Critical files:**
- `src/hssm/addm/__init__.py` (new) — re-export `aDDM` and `aDDMConfig`, mirroring [src/hssm/rl/__init__.py](data/azhang/HSSM/src/hssm/rl/__init__.py).
- [src/hssm/__init__.py](data/azhang/HSSM/src/hssm/__init__.py) — add `from .addm import aDDM` (line 22 already has `from .rl import RLSSM`) and add `"aDDM"` to `__all__`.

> **Architecture update:** `aDDM` follows the **`RLSSM` registration pattern, not the `HSSM(model=...)` pattern**. Like `RLSSM`, it is *not* added to `default_model_config` and *not* registered through `register_model`. Users instantiate it directly:
> ```python
> import hssm
> model = hssm.aDDM(data=addm_trial_df, model_config=cfg, ...)
> ```
> No `src/hssm/modelconfig/addm_config.py` is created — that directory holds dicts consumed by `register_model` for the analytical/ONNX-based `HSSM(model="ddm", ...)` flow, which doesn't apply to subclass-based families like RLSSM and aDDM.

**Reuses:** `hssm/__init__.py` re-export pattern (see existing `RLSSM` import at line 22).

### Step 7 — Data validation for aDDM-specific columns

**Critical file:** [src/hssm/data_validator.py](data/azhang/HSSM/src/hssm/data_validator.py).

The DataValidatorMixin currently validates that `extra_fields` columns exist ([line 46](data/azhang/HSSM/src/hssm/data_validator.py#L46)). aDDM also needs **shape validation** because `sacc_array` is a 2D array-of-arrays stored inside a DataFrame column.

Add an optional `_validate_addm_columns()` method invoked when `model_config.model_name == "addm"`:
- `r1`, `r2`, `d`, `flag` must be 1D numeric with length `n_trials`.
- `sacc_array` must be a 2D array of shape `(n_trials, max_d)` (or a column of variable-length lists, padded internally via `pad_sacc_array_safely`).
- `d[i] <= sacc_array.shape[1]`.

Minimally invasive: put the hook in `_post_check_data_sanity` and no-op for non-aDDM models.

### Step 8 — Tests

Create a `tests/addm/` directory mirroring [tests/rl/](data/azhang/HSSM/tests/rl/):

- `tests/addm/test_addm_config.py` — patterned after [tests/test_rlssm_config.py](data/azhang/HSSM/tests/test_rlssm_config.py).
- `tests/addm/test_addm.py` — patterned after [tests/test_rlssm.py](data/azhang/HSSM/tests/test_rlssm.py).
- `tests/addm/test_addm_builder_output_shape.py` — patterned after [tests/test_rl_builder_output_shape.py](data/azhang/HSSM/tests/test_rl_builder_output_shape.py).
- `tests/addm/test_addm_likelihood.py` — patterned after [tests/test_rldm_likelihood.py](data/azhang/HSSM/tests/test_rldm_likelihood.py).

Test classes:

1. `TestaDDMConfigCreation` — build `aDDMConfig`, assert defaults, assert `validate()` raises on missing/inconsistent fields.
2. `TestaDDMConfigFromDefaults` — assert `aDDMConfig.from_defaults(...)` raises `NotImplementedError` (mirrors RLSSMConfig).
3. `TestaDDMLikelihood` — tiny synthetic dataset (10 trials), confirm `logp` is finite, gradient w.r.t. each parameter is finite, matches a direct call to the vendored `get_addm_fptd_jax_fast`.
4. `TestaDDMEndToEnd` — 200-trial synthetic dataset, `hssm.aDDM(data=..., model_config=...)` builds, a single MCMC draw succeeds (smoke test, `draws=5, tune=5`).

> Removed: the obsolete `TestaDDMConfigConversion` (`.to_config()` round-trip) — there is no `to_config()` method in the post-rebase architecture.

**Reuse:** test fixtures from `tests/conftest.py`.

### Step 9 — Tutorial notebook

Create `docs/tutorials/addm_tutorial.ipynb` mirroring the structure of `docs/tutorials/rlssm_tutorial.ipynb`:
- Load/simulate a small aDDM dataset (reuse `simulate_addm` from efficient-fpt example6).
- Build the model with `hssm.aDDM(data=..., model_config=aDDMConfig(...))` — **not** `hssm.HSSM(model="addm", ...)`, since aDDM follows the RLSSM subclass pattern.
- Add a hierarchical regression on `eta` (e.g., by participant) to showcase why using HSSM buys more than raw efficient-fpt.
- Run `model.sample()` and plot posteriors via `arviz`.

### Step 10 — Cleanup

- Delete or rename the stale [addm_andrew_dev](data/azhang/HSSM/addm_andrew_dev) folder (it has a trailing space in its name, which is a foot-gun on many filesystems) once the new module is working.
- Update `README.md` example list and `mkdocs.yml` nav to include the new tutorial.

---

## Files to be created

**HSSM-original code (mirrors `src/hssm/rl/` layout post-rebase):**
- `src/hssm/addm/__init__.py` — re-exports `aDDM`, `aDDMConfig`
- `src/hssm/addm/config.py` — `aDDMConfig` dataclass *(peer of `hssm/rl/config.py`)*
- `src/hssm/addm/addm.py` — `aDDM(HSSMBase)` class *(peer of `hssm/rl/rlssm.py`)*
- `src/hssm/addm/utils.py` — `validate_addm_panel` *(peer of `hssm/rl/utils.py`)*
- `src/hssm/addm/attention_process.py` — `standard_alternating` and registry
- `src/hssm/addm/likelihoods/__init__.py`
- `src/hssm/addm/likelihoods/builder.py` — `make_addm_logp_func`, `make_addm_logp_op`
- `src/hssm/addm/likelihoods/addm_jax.py` — thin wrapper composing attention process + vendored JAX likelihood
- `tests/addm/test_addm_config.py`
- `tests/addm/test_addm.py`
- `tests/addm/test_addm_builder_output_shape.py`
- `tests/addm/test_addm_likelihood.py`
- `docs/tutorials/addm_tutorial.ipynb`

**Vendored from efficient-fpt (verbatim copies, kept in their own subpackage):**
- `src/hssm/addm/likelihoods/jax/__init__.py`
- `src/hssm/addm/likelihoods/jax/multi_stage.py` ← `efficient_fpt_jax/multi_stage.py`
- `src/hssm/addm/likelihoods/jax/single_stage.py` ← `efficient_fpt_jax/single_stage.py`
- `src/hssm/addm/likelihoods/jax/utils.py` ← `efficient_fpt_jax/utils.py`
- `src/hssm/addm/likelihoods/jax/NOTICE` (or LICENSE) — upstream attribution and commit hash

## Files to be modified

- `src/hssm/__init__.py` — `from .addm import aDDM` and add `"aDDM"` to `__all__` (mirrors the existing `RLSSM` import on line 22).
- `src/hssm/data_validator.py` — add aDDM column-shape validation hook.
- `README.md`, `mkdocs.yml` — mention the new model.

> **Removed from "files to be modified":**
> - ~~`src/hssm/config.py`~~ — `aDDMConfig` lives in its own subpackage at `src/hssm/addm/config.py`, not in the central `config.py`. (RLSSMConfig was moved out of `hssm.config` in the rebase for the same reason.)
> - ~~`src/hssm/defaults.py`~~ — aDDM is **not** registered in `default_model_config`; users instantiate `hssm.aDDM(...)` directly, same as `hssm.RLSSM(...)`.
> - ~~`pyproject.toml`~~ — no new dependencies (JAX is already core).

## Key functions/utilities to reuse (no re-implementation)

| Purpose | Location |
|---|---|
| JAX FPT likelihood | `hssm.addm.likelihoods.jax.get_addm_fptd_jax_fast` *(vendored)* |
| Safe padding of saccade arrays | `hssm.addm.likelihoods.jax.pad_sacc_array_safely` *(vendored)* |
| Likelihood `Op` builder pattern | [hssm/rl/likelihoods/builder.py](data/azhang/HSSM/src/hssm/rl/likelihoods/builder.py) |
| Subclass `__init__` / `_make_model_distribution` pattern | [hssm/rl/rlssm.py](data/azhang/HSSM/src/hssm/rl/rlssm.py) |
| Family-specific config dataclass pattern | [hssm/rl/config.RLSSMConfig](data/azhang/HSSM/src/hssm/rl/config.py) |
| Abstract base for HSSM model classes | [hssm/base.HSSMBase](data/azhang/HSSM/src/hssm/base.py#L92) |
| `make_distribution` (consumes pre-built loglik `Op`) | `hssm.distribution_utils.make_distribution` |
| Extra-fields propagation into logp | [data_validator.DataValidatorMixin._update_extra_fields](data/azhang/HSSM/src/hssm/data_validator.py#L156) |
| Param bound enforcement | [distribution_utils.dist.apply_param_bounds_to_loglik](data/azhang/HSSM/src/hssm/distribution_utils/dist.py#L40) |

---

## Verification

End-to-end checks, in order:

1. **Unit**: `pytest tests/addm/ -v` — all test files pass, including finite-gradient check.
2. **Likelihood parity**: in `tests/addm/test_addm_likelihood.py`, assert HSSM's wrapped op returns the same value (to 1e-6) as a direct call to the vendored `hssm.addm.likelihoods.jax.get_addm_fptd_jax_fast` on a 10-trial fixture. This confirms the HSSM extra-fields/op-wrapping plumbing does not corrupt the underlying JAX computation. (A separate, off-CI sanity script may also compare against an installed `efficient-fpt` checkout to detect drift between the vendored copy and upstream.)
3. **Smoke sample**: `hssm.aDDM(data=synthetic_trials, model_config=cfg).sample(draws=5, tune=5)` completes without error and returns an `InferenceData`.
4. **Parameter recovery**: larger off-CI script (e.g., `tests/scripts/addm_recovery.py`) — simulate 1000 trials with known `(eta, kappa, a, b, x0, sigma)`, fit via `hssm.aDDM(...)`, confirm posterior means within ~2σ of ground truth. Reuse the recovery setup from [efficient-fpt example8_empirical/parameter_recovery.ipynb](data/azhang/efficient-fpt/examples/example8_empirical).
5. **Tutorial runs clean**: `jupyter nbconvert --execute docs/tutorials/addm_tutorial.ipynb` finishes without errors.
6. **Docs build**: `mkdocs build` succeeds with the new tutorial in nav.

## Open questions for the user

1. ~~**Subclass vs config-only**~~ — *resolved by the rebase.* The new architecture (`HSSMBase` + `RLSSM` subclass) settles this: aDDM follows the subclass pattern, with both `aDDM(HSSMBase)` and `aDDMConfig(BaseModelConfig)` introduced.
2. **Non-decision time `t`**: include in v1 as an additional sampled parameter (shift RTs), or defer?
3. **Attention-process extensibility**: is the default `standard_alternating` enough, or should v1 already expose user-pluggable attention processes (e.g., non-alternating fixation patterns)?
4. **Hierarchical regression target**: should the tutorial demonstrate hierarchical regression on `eta` (attention bias) by participant, or is a different parameter (e.g., `kappa`) more meaningful as a worked example?
