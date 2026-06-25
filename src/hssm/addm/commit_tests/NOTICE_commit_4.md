# Commit 4 — `aDDM(HSSMBase)` subclass + data validation + top-level export

Central wiring commit: `hssm.aDDM(data=..., model_config=aDDMConfig(...))` now
constructs and samples. Mirrors `RLSSM` (build the differentiable Op in
`__init__`, inject via `dataclasses.replace(loglik=op, backend="jax")`, override
`_make_model_distribution`) but **without** the balanced-panel reshape and
**without** rejecting `missing_data`/`deadline`.

## What was added / changed

| File | Change |
|---|---|
| `src/hssm/addm/addm.py` | NEW — `class aDDM(HSSMBase)`: `__init__`, `_make_model_distribution`, `_update_extra_fields` override, data validation/materialization helpers. |
| `src/hssm/addm/__init__.py` | Export `aDDM`, `aDDMConfig`, attention-process registry. |
| `src/hssm/__init__.py` | Top-level export of `aDDM`, `aDDMConfig` (peers of `RLSSM`). |
| `src/hssm/addm/likelihoods/builder.py` | Reduce `sigma` to a scalar in `logp` (integration fix — see below). |

`aDDMConfig()` is the default config (per the get_defaults decision — **not**
`aDDMConfig.get_defaults()`). Fresh `list()` copies are passed to
`make_addm_logp_op` so HSSMBase appending `"p_outlier"` is invisible to the Op.

## Integration fixes discovered while wiring (not in the original sketch)

The HSSM/extra-fields machinery assumes scalar-per-row covariates; the aDDM
kernel wants scalar core params + per-trial covariates with a 2-D `sacc_array`.
Three concrete mismatches were found and fixed:

1. **`sacc_array` as an object column crashes bambi.** `bmb.Model` runs
   `with_categorical_cols`, which `astype("category")` every object column;
   array cells are unhashable → factorize crash. Fix: `_prepare_addm_data`
   converts `sacc_array` to **padded hashable tuples** before `super().__init__`
   (bambi categorizes its own copy harmlessly; `self.data` keeps the tuples).
   `_make_model_distribution` and an overridden `_update_extra_fields` both stack
   the column back to a `(n_trials, max_d)` float array via `_stack_sacc_array`.
   (`_update_extra_fields` runs inside `sample()`, so the override is required
   here, not deferrable to Commit 6.)

2. **Core params must stay scalar.** Unlike RLSSM (whose kernel takes per-trial
   params), the aDDM kernel vmaps over trials but treats `eta, kappa, a, b, x0`
   as scalars. `param.is_trialwise` is `True` here, so the HSSM-style
   `params_is_trialwise` broadcast them to `(n_obs,)` and the drift build
   collapsed (`(6,)` vs `(200,)`). Fix: `params_is_trialwise = all False`.
   Consequence: per-trial / **regression** core params are not yet supported by
   the kernel — such models *build* (test 7) but should not be sampled.

3. **`sigma` must be scalar.** It is a model-level diffusion constant; passing
   the `(n_obs,)` column collided with the `(order,)` quadrature grid. Fix: the
   builder reduces `sigma` to a scalar (`jnp.asarray(sigma).reshape(-1)[0]`); the
   validator requires the `sigma` column constant across trials.

4. **Post-hoc `log_likelihood` (WAIC/LOO) unsupported.** HSSM's pointwise
   log-likelihood re-evaluates the Op with a *draw* dimension on the params,
   which the scalar-param kernel can't broadcast. The smoke test samples with
   `idata_kwargs={"log_likelihood": False}`. Supporting it needs the kernel to
   carry a sample dimension — a separate enhancement.

`missing_data`/`deadline` are allowed (forwarded to the base class). Note: their
interaction with the baked row order / `sacc_array` materialization is untested
(no within-participant ordering to corrupt, but PPC-side coordinate handling is
Commit 6).

## Tests — `test_commit4_subclass.py`

Run in the repo venv (`source .venv/bin/activate`), synthetic data built with
numpy (no `efficient_fpt`); `rt ∈ (0.05, 0.45)` to stay in-support for the
default `b=2.0` (bound collapses near `t≈a/b`).

1. `test_construct_default_config` — `hssm.aDDM(data=df)` constructs.
2. `test_construct_explicit_config` — with explicit `aDDMConfig()`.
3. `test_is_hssmbase_subclass` — subclass + `.sample/.bounds/.model`.
4. `test_loglik_op_injected` — Op + `backend=="jax"`; caller config untouched.
5. `test_bad_columns_raise` — non-binary flag / `d>width` / missing column raise.
6. `test_smoke_sample` — 200 trials, finite init logp, `.sample(draws=5, tune=5,
   chains=1, cores=1, idata_kwargs={"log_likelihood": False})` → `InferenceData`.
7. `test_hierarchical_regression_builds` — `include=[{eta ~ 1 + (1|participant)}]`
   builds.

**Status:** all 7 pass in the repo venv.
