# Commit 3 — `aDDMConfig` dataclass

Adds the config object that names the sampled parameters, per-trial covariates,
bounds, and the attention process — the data the `aDDM` subclass (Commit 4) will
consume. Mirrors `RLSSMConfig` on `BaseModelConfig`. aDDM is **not** a
`SupportedModels` entry; the config is constructed directly.

## What was added

| File | Purpose |
|---|---|
| `src/hssm/addm/config.py` | `@dataclass aDDMConfig(BaseModelConfig)` with `validate()`, `get_defaults(self, param)`, and `from_addm_dict()`. |

`addm/__init__.py` is intentionally **not** touched — exporting `aDDMConfig` at
the package/top level is Commit 4.

## Design notes

- **Every field defaults**, so `aDDMConfig()` is a complete default config and no
  `kw_only=True` is needed (`model_name="addm"`, etc.). This is the factory the
  subclass uses in Commit 4 (`model_config or aDDMConfig()`).
- `loglik` / `backend` are inherited as `None` and injected by `aDDM.__init__`
  via `dataclasses.replace` in Commit 4 — not redeclared here.
- `get_defaults(self, param)` overrides the **abstract** `BaseModelConfig`
  method (same contract as `RLSSMConfig`: returns `(prior=None, bounds)`).
  Plan-sketch's no-arg `get_defaults()` factory was dropped (it would collide
  with the abstract per-param method); confirmed with the user.
- `validate()` checks: `list_params` present, `attention_process` resolves
  (via `resolve_attention_process`), `params_default` length matches
  `list_params`, every `list_params` entry has a `bounds` entry, and
  `extra_fields` is present.

## Plan-sketch fixes (deviations, both corrections)

- The sketch's `params_default = [0.3, 1.0, 1.0, 2.0, 0.0, 0.0]` had **6 values
  for 5 params** (would fail `validate()`'s length check) and `b = 2.0` lies
  **outside** its `b:(0.0, 1.0)` bound. Replaced with a valid 5-value default
  `[0.3, 1.0, 1.0, 0.25, 0.0]` (eta, kappa, a, b, x0 — all in-bounds). The test
  asserts every default sits inside its bound.

## Tests — `test_commit3_config.py`

Run in the repo venv (`source .venv/bin/activate`):

```
JAX_PLATFORMS=cpu python src/hssm/addm/commit_tests/test_commit3_config.py
```

1. `test_defaults` — expected fields; params_default valid + in-bounds.
2. `test_instantiable_and_subclass` — subclass of `BaseModelConfig`; constructs
   (proves both abstract methods are overridden).
3. `test_validate_ok` — default config validates.
4. `test_validate_rejects_unknown_attention_process`.
5. `test_validate_rejects_bounds_param_mismatch`.
6. `test_validate_rejects_params_default_mismatch`.
7. `test_get_defaults` — `("eta") -> (None, (0.0, 1.0))`; unknown -> `(None, None)`.
8. `test_from_addm_dict_roundtrip` — round-trips and drops unknown keys.

**Status:** all 8 pass in the repo venv.
