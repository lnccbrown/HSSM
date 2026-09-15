# Issue #1305 — bambi 0.20 migration: fix plan

Companion to [`bambi-dev-upgrade-test-failures.md`](./bambi-dev-upgrade-test-failures.md),
which inventories *what* fails and *why*. This document is the *what to do
about it* list: nine work items, each sized to one PR.

**Upstream change:** [bambinos/bambi#1002 "Bambi rewrite"](https://github.com/bambinos/bambi/pull/1002)
(merged 2026-09-07) — a frontend/backend split, not a feature release.
**Tracking parent:** #1306.

## Current state

After F1 (#1310) and F2 (#1311):

```
1291 passed, 4 skipped, 124 xfailed, 1 xpassed, 0 failed
```

The 102 ids that still fail are marked
`@pytest.mark.xfail(reason="bambi 0.20 migration (#1305): R<n> …", strict=False)`.
Every R1 mark is gone; the ids R1 was masking were re-triaged into R2, R5 and
three new root causes (R10-R12) — see *Re-triage after F1* in the inventory.
Marks are per function except where a parametrized test is only partly
affected (`test_vi_matrix`, `test_missing_data_vi_matrix`, the two
`test_plot_model_cartoon_*_choice` grids), which carry per-row
`pytest.param(..., marks=...)`.

## Fix items

All nine are tracked as sub-issues of #1306.

| # | Issue | Fix | Root causes | Unblocks | Risk |
|---|-------|-----|-------------|---------:|------|
| F1 | #1310 | ~~Declare `SSMFamily.RESPONSE_NDIM = 2`, drop dead hook, fix coord name~~ **done** | R1 | 253 ids | Low |
| F2 | #1311 | Collapse `hssm.Link` onto `inverse_link` | R3, R4 | 11 ids + 3 collection errors | Medium (user-facing) |
| F3 | #1312 | Let truncated-prior callables accept `dims` | R2 | 9 ids | Low |
| F4 | #1313 | Replace removed likelihood-parameter APIs | R5 + latent #1 | 6 ids | Medium |
| F5 | #1314 | Migrate deprecated accessors, refresh test doubles | R6 + latent #4 | 3 ids | Low |
| F6 | #1315 | Handle the new `predictions` DataTree group | latent #2 | 0 (masked) | High (behavioral) |
| F7 | #1316 | Adapt to new-group prediction semantics | latent #5 | 0 (masked) | High (behavioral) |
| F8 | #1317 | Reconcile two drifted assertions | R7, R9 | 2 ids | Low |
| F9 | #1318 | Investigate numba slice-sampler `SystemError` | R8 | 2 ids | Unknown (likely upstream) |
| F10 | #1328 | VI on the JAX compile backend cannot trace the symbolic `__obs__` alloc | R11 | 0 left | Done |
| F11 | #1329 | aDDM posterior predictive: pymc forward sampler rejects a `TensorConstant` | R12 | 3 ids | Medium |
| F12 | #1330 | Drop bambi's constant-parameter and `*_Intercept_centered` posterior variables | R13 | 5 ids | Low |

F6 (#1315) is no longer latent: R10 (20 ids) is the `predictions` group
surfacing in every plotting path that calls `sample_posterior_predictive(data=...)`.
F4 (#1313) grew from 6 ids to 51 — R5 sits on every `sample()` and `find_MAP`
path, so it was masked almost as completely as R1.

Ordering matters: **F1 first**. It aborts model construction, so it masks
almost everything else. F6 and F7 are latent behavioral changes that F1 will
expose — expect the failure count to rise on the first re-run after F1 lands.

---

### F1 (#1310) — Declare the SSM response dimensionality

**Root cause:** R1 (253 ids, 28 files, 123 xfail marks)

`SSMFamily` overrides `create_extra_pps_coord`, the pre-rewrite hook, which no
longer exists in bambi and is now dead code. Families describe their response
through `RESPONSE_NDIM` instead.

The threshold is subtle: `build_response_term` appends the response coord when
`RESPONSE_NDIM > 0`, but `coords_from_response` only *creates* that coord when
`RESPONSE_NDIM > 1`. `RESPONSE_NDIM = 1` fixes nothing — it must be **2**.

Verified by monkeypatch (no source change): a plain DDM builds successfully and
the model coords become `['__obs__', 'rt_dim']`.

- [x] `src/hssm/distribution_utils/dist.py:753` — set `RESPONSE_NDIM = 2`
- [x] `src/hssm/distribution_utils/dist.py:756` — remove `create_extra_pps_coord`
- [x] `src/hssm/base.py:1287`, `src/hssm/base.py:1356` — the hardcoded
      `"rt,response_extra_dim_0"` coord is now `rt_dim`; both checks folded
      into `_rename_response_dim`
- [x] Re-run the suite and re-triage; remove the R1 xfail marks that now pass
      (all 142 R1 marks removed; 102 ids re-marked as R2/R5/R10/R11/R12)

### F2 (#1311) — Collapse `hssm.Link` onto a single `inverse_link`

**Root cause:** R3 (7 ids + 3 collection errors), R4 (4 ids)

PR #1002 *Custom links*: "a single `inverse_link` function compatible with the
PyMC/PyTensor backend is used instead of the `linkinv` and `linkinv_backend`
functions… The forward `link` function is optional." The signature is now
`Link(name, link=None, inverse_link=None)`.

This is **not a rename** — bambi deliberately dropped the NumPy/PyTensor split
that `hssm.Link` is built around, so HSSM's public signature and docs change.
Mitigating detail: `_make_generalized_sigmoid_simple` already uses `np.exp`,
which dispatches correctly on PyTensor tensors, so HSSM's two inverses are
already the same function.

- [ ] `src/hssm/link.py:69-95` — replace `linkinv`/`linkinv_backend` with
      `inverse_link` in both the `HSSM_LINKS` branch and the `super().__init__`
      call
- [ ] `src/hssm/link.py:21-67` — update the docstring and the `custom_log`
      example
- [ ] Note the breaking change in `docs/changelog.md`
- [ ] Fixes the 3 collection errors (`tests/unit/test_prior.py`,
      `tests/unit/param/test_regression_param.py`,
      `tests/unit/param/test_unmatched_group_prior_graph.py`), which have no
      xfail marks because an import-time `TypeError` leaves pytest no test item
- [ ] Remove the R3/R4 xfail marks

### F3 (#1312) — Let truncated-prior callables accept `dims`

**Root cause:** R2 (9 ids, 4 files)

Intercept uncentering moved into the PyMC graph (PR *Predictor centering and
the intercept*; `Model._re_center_intercept` is gone), so bambi now builds a
`_centered` RV and passes `dims=` when instantiating a prior's distribution.
HSSM's `TruncatedDist(name)` closure takes name only.

- [x] `src/hssm/prior.py:168-180` — accept and forward `dims` (and `shape`):
      `TruncatedDist(name, **call_kwargs)` merges the call-time kwargs over the
      closure-time `pymc_dist_args` before calling `pm.Truncated`
- [x] Confirm behaviour for both bounded and unbounded priors, and for
      regression intercepts specifically (unit tests in `tests/unit/test_prior.py`;
      `test_identity_safe_prior_graph.py` now compares the free
      `a_Intercept_centered` RV, since `a_Intercept` is a Deterministic)
- [x] `src/hssm/base.py` `_postprocess_initvals_deterministic` — look the
      `INITVAL_SETTINGS` default and the user-supplied `initval` up by the
      name with `_centered` stripped, so `a_Intercept_centered` (the RV old
      bambi fit as `a_Intercept`) still receives the `a_Intercept` default
- [x] Remove the R2 xfail marks (the one on
      `tests/test_sample_posterior_predictive.py` was re-marked as R13 — its
      pre-sampled fixture lacks `v_Intercept_centered`, which is #1330)

### F4 (#1313) — Replace the removed likelihood-parameter APIs

**Root cause:** R5 (6 ids) + latent #1

Two deletions, one job. PR #1002 removes `Model._compute_likelihood_params` and
lists `Family._make_dist_kwargs_and_coords` by name: "distribution arguments and
coordinates are no longer reconstructed from posterior samples."

The PR names the replacement under *Graph evaluation and interventions*:
"Conditional parameters are represented as deterministic variables in the PyMC
model. The backend can evaluate them with `pymc.compute_deterministics`."

- [x] `src/hssm/base.py:1028`, `src/hssm/base.py:1034` —
      `_compute_likelihood_params` → `Model.predict(kind="response_params")`
- [x] `src/hssm/utils.py:205` — `_compute_likelihood_params` → HSSM-owned
      `_compute_likelihood_params` wrapping `predict`, folding the out-of-sample
      `predictions` group back into one dataset
- [x] `src/hssm/utils.py:286` — `_make_dist_kwargs_and_coords` ported into
      `hssm.utils` on top of `Model.parameters`
- [x] Remove the R5 xfail marks (all 19). The 2 R8 ids under `test_choice_only`
      now carry their own mark (see F9); 5 ids re-surfaced as R13 (F12)

### F5 (#1314) — Migrate deprecated accessors and refresh test doubles

**Root cause:** R6 (3 ids) + latent #4

The rewrite renamed the model vocabulary, with `FutureWarning` shims that will
be removed:

| Old | New |
|-----|-----|
| `Model.components` | `Model.parameters` |
| `Model.distributional_components` | `Model.conditional_parameters` |
| `Model.constant_components` | `Model.marginal_parameters` |
| `Model.response_component` | `Model.response_term` |

`src/hssm/utils.py` is currently inconsistent: line 55 uses the new
`response_term`, lines 280 and 285 still use `response_component.term`.

- [x] `src/hssm/utils.py:134`, `src/hssm/base.py:1812` — `components`
- [x] `src/hssm/utils.py:127`, `:142`, `src/hssm/base.py:915` —
      `distributional_components`
- [x] `src/hssm/utils.py:280`, `:285` — `response_component`
- [x] `tests/test_utils.py` — the `SimpleNamespace` doubles still mimic the
      pre-rewrite API
- [x] Fail the test suite on bambi's deprecation `FutureWarning`s
      (`filterwarnings` in `pyproject.toml`, matched on the message — the
      shims warn with `stacklevel=2`, so a module filter on `bambi` never fires)
- [x] Remove the R6 xfail marks (none were left by the time F5 landed)

### F6 (#1315) — Handle the new `predictions` DataTree group

**Root cause:** latent #2 — masked by R1, no failing test yet

PR #1002 *Prediction organization*: when `data` is passed to `Model.predict`,
results land in `predictions` (and `predictions_constant_data`), **not**
`posterior_predictive`. `src/hssm/base.py:1193` reads
`dt_copy["posterior_predictive"]` unconditionally after predicting with `data`,
which will `KeyError` on out-of-sample prediction. Nearby lines in
`sample_posterior_predictive` share the assumption.

This needs **new test coverage**, not just a rename — there is currently no
failing test to tell you when it is fixed.

- [x] Audit `src/hssm/base.py:1039-1214` for the group assumption — four
      reads of `dt_copy["posterior_predictive"]` (safe-mode chunking, the
      non-safe in-place copy, and the two non-in-place returns). Collapsed
      into one `_pop_response_draws` helper and a single assembly path; the
      non-safe, non-in-place branch now also restores the full posterior on
      the returned copy, like the safe-mode branch always did.
- [x] Decide whether HSSM surfaces `predictions` or normalises it back into
      `posterior_predictive` for API stability. **Decision:** normalise.
      Only the response variable is moved into `posterior_predictive`;
      `predictions` / `predictions_constant_data` are removed. The trial-wise
      parameters bambi bundles into `predictions` are dropped (they were never
      part of HSSM's contract, and keeping them would give `posterior_predictive`
      a different variable set in and out of sample). `kind="response_params"`
      stays a thin pass-through to bambi, as #1313 already relies on.
- [x] Add out-of-sample prediction tests — `tests/test_sample_posterior_predictive.py`
      fits a small regression DDM (the `cavanagh_idata.nc` fixture predates
      bambi 0.20 and lacks `v_Intercept_centered`, so in-sample `predict` on it
      raises `KeyError`) and checks group layout, `__obs__` size, stale-group
      cleanup, and in-/out-of-sample parity across `safe_mode` × `inplace`.
- [x] Remove the R10 xfail marks. 4 of the 20 pass (`test_quantile_probability`,
      `test_predictive`). The 16 in `tests/test_plotting_cartoon.py` now fail
      one step later: `idata_cavanagh_cartoon.nc` also predates bambi 0.20, so
      `pm.compute_deterministics` cannot find `v_Intercept_centered`. They stay
      xfail with that reason — **regenerating the `.nc` fixtures is tracked in
      #1336.** The cartoon path was verified end-to-end on a fresh trace.

### F7 (#1316) — Adapt to the new new-group prediction semantics

**Root cause:** latent #5 — masked by R1, no failing test yet

`sample_new_groups` is deprecated and **has no effect** (`bambi/models.py:1028`).
bambi now infers the strategy from the grouping value: null (`None`, `np.nan`,
`pd.NA`) means unknown identity and samples among observed groups, while a
non-null unseen value means a genuinely new group whose coefficients are drawn
from the population-level model.

`src/hssm/utils.py:195` sets the flag and `:206` passes it. Nothing raises — the
semantics of HSSM's hierarchical prediction simply changed.

- [ ] Remove the dead `sample_new_groups` plumbing
- [ ] Document the new semantics for HSSM users
- [ ] Add tests covering both the unknown-identity and new-group paths

### F8 (#1317) — Reconcile two drifted assertions

**Root causes:** R7 (1 id), R9 (1 id)

- **R7** — `tests/test_hssm.py::test_transform_params_general[include4-IndexError]`
  expects `IndexError` for a malformed formula; the bumped `formulae` stack
  looks the name up as a data-frame column, so pandas raises `KeyError` first.
  Worth deciding whether HSSM should normalise this into its own `ValueError`
  rather than chase the upstream type.
- **R9** — `tests/test_noncentered.py::test_unknown_dict_key_raises_at_construction`
  expects `[Uu]nknown component`; HSSM raises `"Unknown parameter name(s) in
  \`noncentered\`: [...]"`. Still a `ValueError`, only the wording moved.
  **Nothing in PR #1002 touches this** — it is a stale assertion against HSSM's
  own code and is probably independent of #1305.

- [x] Decide intended behaviour for each, then update assertion or code
  - R7: HSSM now wraps formulae/pandas lookup errors from `design_matrices`
    in its own `ValueError` naming the parameter and formula
    (`RegressionParam._get_design_matrices`), so the type no longer
    depends on the upstream stack.
  - R9: the check lives in bambi; the assertion now tests behaviour (bad
    key + valid names reported) instead of bambi's wording.
- [x] Remove the R7/R9 xfail marks

### F9 (#1318) — Investigate the numba slice-sampler `SystemError`

**Root cause:** R8 (2 ids)

```
SystemError: CPUDispatcher(<function numba_funcified_fgraph ...>) returned a
result with an exception set
  .../pymc/step_methods/slicer.py: while y <= logp(ql)
```

Both `tests/integration/test_choice_only.py::test_choice_only[*-pymc-slice-beta_reg]`
cases die inside the numba-compiled logp. **No bambi code appears in the
traceback** — this comes from the transitive `pymc`/`pytensor` major bumps that
PR #1002 pulls in, not from bambi's own API.

Corroborating: the PR adds `pytest-forked` to bambi's own test dependencies
"because it is used in manual testing to prevent the numba backend from
exhausting the available RAM", and drops the `numba` pin from the `nutpie`
extra. Upstream is aware this path is fragile.

- [ ] Reproduce against the pre-upgrade dependency set to confirm attribution
- [ ] If it reproduces on stock pymc/pytensor, report upstream rather than
      working around it in HSSM
- [ ] These 2 ids sit under `test_choice_only`, which is marked R5 — after F4
      lands they will need their own mark or a fix

### F10 (#1328) — VI on the JAX compile backend cannot trace `__obs__`

**Root cause:** R11 (7 ids) — **resolved**.

bambi 0.20 keeps the data and the `__obs__` dim length as shared variables;
the JAX linker traces shared variables, so every shape derived from them
(bambi's response-parameter broadcast, HSSM's missing-data `n_missing`
slice) is dynamic. `HSSM.vi(backend="jax")` now freezes them to constants
via `pm.fit(more_replacements=...)` (`_vi_compat.freeze_shared_data`), the
same step pymc's JAX samplers take in `get_jaxified_graph`.

- [x] Pin down the nodes (bambi `build.py:93` broadcast; HSSM `dist.py`
      `n_missing`) — both shape-from-shared-variable, no `pm.Data` static
      shape available
- [x] Fix in HSSM; upstream gap is pytensor's `JAXLinker` static-argument
      scan (unfiled)
- [x] Remove the R11 marks

### F12 (#1330) — Drop the extra posterior variables bambi 0.20 records

**Root cause:** R13 (5 ids), unmasked by F4

bambi 0.20 writes constant marginal parameters (`p_outlier=0.05`, a fixed
`z=0.5`, the `0.0` placeholder of a fixed-vector parameter) to `posterior` as
deterministics, and keeps the `*_Intercept_centered` RV alongside the
uncentered intercept. Old bambi stored neither, so `az.summary` gains rows and
fixed-vector parameters show up in the trace.

- [x] Decide whether `_clean_posterior_group` should drop constant
      deterministics and `*_centered` RVs. **Decision:** drop every parameter
      bambi built as a `pm.Deterministic` (the constants), unconditionally —
      it restores the pre-rewrite trace. **Keep** `*_Intercept_centered`: it is
      the actual free RV, and bambi's `predict` / `compute_log_likelihood`
      (`pm.compute_deterministics`) raise `KeyError` without it. Reconstructing
      it on demand the way bambi does for `_offset` would have to wrap every
      `model.predict` entry point, so the extra summary row is accepted.
- [x] Remove the R13 xfail marks; regression-model row counts in
      `test_post_processing_reg*` now include the centered intercept(s)

---

## Definition of done for the migration

- [ ] No `xfail` marks referencing `bambi 0.20 migration (#1305)` remain
- [ ] No collection errors
- [ ] No bambi `FutureWarning`s emitted from `src/hssm/`
- [ ] `uv run prek run --all-files` clean (`ruff`, `pyrefly`, `mypy`)
- [ ] Notebook CI green (`check_notebooks.yml`) — not covered by this inventory
- [ ] Breaking changes recorded in `docs/changelog.md`, `hssm.Link` in particular
