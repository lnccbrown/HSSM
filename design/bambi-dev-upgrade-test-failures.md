# Issue #1305 — bambi dev upgrade: test failure inventory

**Branch:** `1309-migration-mark-all-failed-tests`
**bambi under test:** `0.20.1.dev2+gb0c49d5db` (git `dev` branch)
**HSSM commit at time of run:** `cfbd44e8` (import-layer repairs already applied)
**Upstream change:** [bambinos/bambi#1002 "Bambi rewrite"](https://github.com/bambinos/bambi/pull/1002),
merged 2026-09-07. Every root cause below is cross-referenced against that PR's
release notes; section names in *italics* refer to headings in the PR body.

**Command:**

```bash
uv run pytest tests/ -p no:randomly -o addopts="" -o log_cli=false \
    --continue-on-collection-errors -n 8 --timeout=900 -q
```

## Headline result

```
Before:  250 failed, 673 passed, 5 skipped, 60 errors, 3 collection errors
After:   673 passed,   5 skipped, 286 xfailed, 0 failed, 0 xpassed
```

The `-rf` summary hides setup errors, so the 250 `FAILED` node ids were merged
with the 36 distinct `ERROR at setup of …` ids: **286 failing test ids across
143 functions in 33 files**, all now marked `xfail`. The 24 errors remaining
after marking are the same 3 collection errors, reported once per xdist worker.

## The shape of the upstream change

PR #1002 is not a feature release — it is a **frontend/backend rewrite**. bambi
now keeps a backend-agnostic description of the model, and *all* PyMC/PyTensor
work (building, predicting, log-likelihood, prior sampling) moved into
`bambi.backend.pymc`, which became a package. The practical consequence for
HSSM is that the extension points HSSM was built on no longer exist in the same
place:

- Families are now **descriptions only** (`DATA_TYPE`, `RESPONSE_NDIM`,
  `PARAMETERS`). Behaviour that used to be family methods is registered in the
  backend via `TransformsRegistry.transform_{data,parameters,predictor}`.
- Terms and parameters lost their `build`/`predict` methods and their `coords`
  properties; free functions (`build_intercept_term`, `build_common_term`,
  `build_conditional_parameter`, …) consume the descriptions instead.
- Duplicate NumPy/xarray reimplementations of the model were deleted in favour
  of operating on the PyTensor graph directly (`pm.do`, `pm.set_data`,
  `pm.compute_deterministics`).

HSSM overrides or reaches into all three layers, which is why the blast radius
is what it is.

## Root causes

All 286 fall into **nine** root causes. R1 accounts for 253 of them because it
fires inside `HSSM.__init__`, so it takes down every test that builds a model
regardless of what the test was actually checking.

| Code | Failing ids | Files | One-line cause | In PR #1002? |
|------|------------:|------:|----------------|--------------|
| R1 | 253 | 28 | `SSMFamily` never declares `RESPONSE_NDIM`, so the 2-D response gets 1-D dims | Yes — *Family definitions* |
| R2 | 9 | 4 | Intercept uncentering moved into the graph; priors are now called with `dims=` | Yes — *Predictor centering and the intercept* |
| R3 | 7 (+3 collection errors) | 1 | `Link` takes `inverse_link`, not `linkinv`/`linkinv_backend` | Yes — *Custom links* |
| R5 | 6 | 1 | `Model._compute_likelihood_params` removed | Yes — *Prediction and log-likelihood moved out of families and parameters* |
| R4 | 4 | 3 | bambi reads `Link.inverse_link`; the `gen_logit` branch sets `linkinv` | Yes — *Custom links* |
| R6 | 3 | 1 | test doubles still mimic the pre-rewrite `Model` API | Yes — *Model parameters and components* |
| R8 | 2 | 1 | numba-compiled blackbox slice sampling raises `SystemError` | Indirect — *Dependency … changes* |
| R7 | 1 | 1 | invalid formula now raises `KeyError`, not `IndexError` | Indirect — formulae bumped |
| R9 | 1 | 1 | error-message wording drift in `noncentered` validation | No — HSSM-internal |

---

### R1 — `SSMFamily` never declares its response dimensionality *(253 ids, 28 files)*

```
pymc.exceptions.ShapeError: Length of `dims` must match the dimensions of the
dataset. (actual 1 != expected 2)
  .../bambi/backend/pymc/terms/response.py:111: pm.Data(label, value, dims=dims, ...)
  name = 'rt,response_data'   value.shape = (n, 2)   dims = ('__obs__',)
```

PR #1002 *Family definitions*: "attributes such as `Family.DATA_TYPE`,
`Family.RESPONSE_NDIM`, and `Family.PARAMETERS` describe the response type,
dimensions, and allowed links." `bmb.Family` defaults `RESPONSE_NDIM = 0`, and
`SSMFamily` (`src/hssm/distribution_utils/dist.py:753`) does not override it.
It instead overrides `create_extra_pps_coord`, which was the *old* hook for
exactly this purpose and no longer exists anywhere in bambi — it is now dead
code that silently does nothing.

**The threshold is subtle and matters.** Two different call sites disagree:

- `build_response_term` appends the response coord when `RESPONSE_NDIM > 0`
  (`bambi/backend/pymc/terms/response.py:37`)
- `coords_from_response` only *creates* that coord when `RESPONSE_NDIM > 1`
  **and** `term.ndim > 1` (`bambi/backend/pymc/model.py`)

So `RESPONSE_NDIM = 1` fixes nothing — the dims tuple stays 1-D. It must be
**`RESPONSE_NDIM = 2`**.

**Verified.** Monkeypatching `SSMFamily.RESPONSE_NDIM = 2` (no source change) and
building a plain DDM gives `BUILD OK` with model coords `['__obs__', 'rt_dim']`.

> **Follow-on:** note the coord is named **`rt_dim`**, from `f"{term.name}_dim"`.
> The old `create_extra_pps_coord` produced `rt,response_extra_dim_0`, which
> HSSM still hardcodes at `src/hssm/base.py:1287` and `src/hssm/base.py:1356`.
> Those two lines will need updating in the same change.

Affected (top files): `tests/test_hssm.py` (35), `tests/integration/test_mcmc.py`
(30), `tests/test_plotting_cartoon.py` (30), `tests/rl/test_rlssm.py` (26),
`tests/test_initvals.py` (19), `tests/integration/test_missing_data_mcmc.py` (18),
`tests/test_sample_posterior_predictive.py` (18), and 21 more files.

### R2 — intercept uncentering moved into the graph *(9 ids, 4 files)*

```
TypeError: _make_truncated_dist.<locals>.TruncatedDist() got an unexpected
keyword argument 'dims'
  .../bambi/backend/pymc/terms/intercept.py:32:
      rv = dist(term.label + "_centered", **kwargs, dims=dims)
```

HSSM represents a bounded prior as a *callable* distribution
(`Prior.dist = _make_truncated_dist(...)`, `src/hssm/prior.py:146-181`) whose
closure signature is `TruncatedDist(name)` — name only.

PR #1002 *Predictor centering and the intercept* explains why this now breaks:
"Previously, intercept samples were corrected after fitting… The intercept on
the original scale is now a deterministic variable in the model." Building that
deterministic means bambi instantiates a `_centered` RV inside the graph, and
the new term builders uniformly pass `dims=` (and broadcast prior args to the
coefficient shape) when doing so. `Model._re_center_intercept` is gone.

The closure needs to accept and forward `dims`/`shape`.

Affected: `tests/test_hssm.py` (3), `tests/test_initvals.py` (3),
`tests/addm/test_addm_subclass.py` (2),
`tests/unit/param/test_identity_safe_prior_graph.py` (1).

### R3 — `Link.__init__` signature changed *(7 ids + 3 collection errors)*

```
TypeError: Link.__init__() got an unexpected keyword argument 'linkinv'
  src/hssm/link.py:88
```

PR #1002 *Custom links*, verbatim: "When defining a custom `Link`, a single
`inverse_link` function compatible with the PyMC/PyTensor backend is used
instead of the `linkinv` and `linkinv_backend` functions. We no longer need one
inverse for NumPy computations and another for the graph. The forward `link`
function is optional."

So this is an intentional, documented break, and the signature is now
`Link(name, link=None, inverse_link=None)`. **This is not a rename** — HSSM
currently maintains a deliberate NumPy/PyTensor split (see the docstring at
`src/hssm/link.py:32-41` and the `custom_log` example at `:57-62`), and that
split has to collapse to one PyTensor-compatible callable. HSSM's public
`hssm.Link` signature and docs change as a result, so this is user-facing.

Because HSSM tests build `hssm.Link` objects at module scope, this also produces
**3 collection errors**, which *cannot* be `xfail`-marked — an import-time
`TypeError` leaves pytest with no test item to attach a marker to:

- `tests/unit/test_prior.py`
- `tests/unit/param/test_regression_param.py`
- `tests/unit/param/test_unmatched_group_prior_graph.py`

Affected test ids: all 7 in `tests/unit/test_link.py`.

### R4 — bambi reads `Link.inverse_link` *(4 ids, 3 files)*

```
AttributeError: 'Link' object has no attribute 'inverse_link'
  .../bambi/backend/pymc/parameters/conditional/build.py:44:
      inverse_link = INVERSE_LINKS.get(link.name, link.inverse_link)
```

The same *Custom links* change, hit from the other side. For names in
`HSSM_LINKS` (i.e. `gen_logit`), `hssm.Link.__init__` bypasses `super().__init__`
and sets `self.linkinv` / `self.linkinv_backend` directly
(`src/hssm/link.py:77-86`), so construction succeeds and the failure is deferred
to build time.

R3 and R4 are one fix. Convenient detail: `_make_generalized_sigmoid_simple`
already uses `np.exp`, which dispatches correctly on PyTensor tensors, so the
two HSSM inverses are the same function today — collapsing them should be
mechanical.

Affected: `tests/test_save_load.py` (2), `tests/test_hssm.py` (1),
`tests/test_jitter.py` (1).

### R5 — `Model._compute_likelihood_params` removed *(6 ids, 1 file)*

```
AttributeError: 'Model' object has no attribute '_compute_likelihood_params'
```

PR #1002 *Prediction and log-likelihood moved out of families and parameters*:
`Family.posterior_predictive`, `Family.log_likelihood`, and the
`DistributionalComponent.predict*` family of methods were all deleted, because
"the reconstruction, with NumPy and `xarray`, of term contributions" is gone.
`Model.compute_log_likelihood` now delegates straight to
`self.backend.compute_log_likelihood(...)` (`bambi/models.py:1068`).

The PR names the replacement mechanism under *Graph evaluation and
interventions*: "Conditional parameters are represented as deterministic
variables in the PyMC model. The backend can evaluate them with
`pymc.compute_deterministics` when needed." That is the path HSSM should take.

HSSM call sites: `src/hssm/base.py:1028`, `src/hssm/base.py:1034`,
`src/hssm/utils.py:205`.

Affected: 6 of the 8 ids in `tests/integration/test_choice_only.py`.

### R6 — stale test doubles for the bambi `Model` API *(3 ids, 1 file)*

```
AttributeError: 'types.SimpleNamespace' object has no attribute 'response_term'
```

Test-side, not bambi's. PR #1002 *Model parameters and components* renames the
whole vocabulary, with `FutureWarning` shims in place:

| Old | New |
|-----|-----|
| `Model.components` | `Model.parameters` |
| `Model.distributional_components` | `Model.conditional_parameters` |
| `Model.constant_components` | `Model.marginal_parameters` |
| `Model.response_component` | `Model.response_term` |

`tests/test_utils.py` fakes a model exposing `response_component.term`, but
`src/hssm/utils.py:55` was already updated (in `cfbd44e8`) to read
`model.response_term`. Note the production code is currently **inconsistent**:
`src/hssm/utils.py:280` and `:285` still use `model.response_component.term`.

Affected: `test_compute_log_likelihood_returns_deep_copy`,
`test_compute_log_likelihood_rejects_missing_family`,
`test_compute_log_likelihood_warns_and_replaces_existing_group`.

### R7 — invalid formula raises `KeyError`, not `IndexError` *(1 id)*

`tests/test_hssm.py::test_transform_params_general[include4-IndexError]` asserts
`IndexError` for a malformed formula. PR #1002 bumps `formulae` (among
`pymc`, `pytensor`, `arviz`, `matplotlib`), and under the new stack the bare
name is looked up as a data-frame column, so pandas raises `KeyError` first.
Exception-type drift — worth deciding whether HSSM should normalise this into
its own `ValueError` rather than chase the upstream type.

### R8 — numba blackbox slice sampling raises `SystemError` *(2 ids)*

```
SystemError: CPUDispatcher(<function numba_funcified_fgraph ...>) returned a
result with an exception set
  .../pymc/step_methods/slicer.py: while y <= logp(ql)
```

Both `tests/integration/test_choice_only.py::test_choice_only[*-pymc-slice-beta_reg]`
cases die inside the numba-compiled logp. **No bambi code appears in the
traceback** — this comes from the transitive `pymc`/`pytensor` major bumps that
PR #1002 pulls in, not from bambi's own API.

Corroborating detail: the PR adds `pytest-forked` to bambi's own test deps
"because it is used in manual testing to prevent the numba backend from
exhausting the available RAM", and drops the `numba` pin from the `nutpie`
extra. Upstream is clearly aware the numba path is fragile in this dependency
set. Treat R8 as upgrade-related but **not** an HSSM/bambi API issue.

### R9 — error-message wording drift *(1 id)*

`tests/test_noncentered.py::test_unknown_dict_key_raises_at_construction`
expects `[Uu]nknown component` but HSSM raises `"Unknown parameter name(s) in
`noncentered`: [...]"`. Still a `ValueError`, only the wording moved. Nothing in
PR #1002 touches this — it is a stale assertion against HSSM's own code and is
almost certainly unrelated to #1305.

---

## Latent breakages — masked by R1, not yet visible in the suite

R1 aborts model construction, so these never get a chance to fail. Cross-
referencing PR #1002 against HSSM's source turns them up statically. **Expect
the failure count to rise, not fall, on the first re-run after R1 is fixed.**

1. **`Family._make_dist_kwargs_and_coords` is explicitly removed.**
   `src/hssm/utils.py:286` calls it. The PR lists it by name under *Family
   definitions*: "distribution arguments and coordinates are no longer
   reconstructed from posterior samples." Same root cause as R5. `pyrefly`
   already flags this line.

2. **`Model.predict(data=...)` writes to a different DataTree group.**
   PR *Prediction organization*: with `data` passed, results land in
   `predictions` (and `predictions_constant_data`), *not*
   `posterior_predictive`. `src/hssm/base.py:1193` reads
   `dt_copy["posterior_predictive"]` unconditionally after calling predict with
   `data`, which will `KeyError` on out-of-sample prediction. Several nearby
   lines in `sample_posterior_predictive` have the same assumption.

3. **Hardcoded response coord name.** `src/hssm/base.py:1287` and `:1356` check
   for `"rt,response_extra_dim_0"`. The rewrite names it `rt_dim` (verified
   above). These checks currently just silently never match.

4. **Deprecated accessors still in production code**, each emitting a
   `FutureWarning` today and scheduled for removal:
   - `model.components` — `src/hssm/utils.py:134`, `src/hssm/base.py:1812`
   - `model.distributional_components` — `src/hssm/utils.py:127`, `:142`,
     `src/hssm/base.py:915`
   - `model.response_component` — `src/hssm/utils.py:280`, `:285`

5. **`sample_new_groups` is deprecated and has no effect** (`bambi/models.py:1028`).
   `src/hssm/utils.py:195` sets it and `:206` passes it. bambi now infers the
   strategy from whether the grouping value is null (unknown identity) vs.
   merely unseen (new group) — a *behaviour* change, so HSSM's hierarchical
   prediction semantics shift even though nothing raises.

6. **`Model.fit(inference_method=...)` now defaults to `None`**, meaning PyMC
   auto-selects a sampler and may pick `nutpie` if installed. `src/hssm/base.py:761-773`
   passes an explicit sampler, so this is currently safe — but any code path
   relying on the old `"pymc"` default would silently change sampler.

**Checked and clear:** the new "`c(...)` response is deprecated, use
`counts(...)`" warning (`bambi/models.py:625`) is guarded to `Multinomial` and
`DirichletMultinomial` families only, so HSSM's `c(rt, response)` with
`SSMFamily` is unaffected.

---

## How the marks were applied

- 142 test functions that fail for **every** parameter set carry a
  function-level decorator:

  ```python
  @pytest.mark.xfail(reason="bambi 0.20 migration (#1305): R1 …", strict=False)
  ```

- `tests/test_hssm.py::test_transform_params_general` is the only test that
  fails for some parameter sets and passes for others (3 of 5). Its three
  failing cases are wrapped in `pytest.param(..., marks=...)` individually; the
  two passing cases are untouched.

- `strict=False` throughout, so a test that starts passing as fixes land shows
  up as `XPASS` rather than turning the suite red. Grep for
  `bambi 0.20 migration (#1305)` to find every mark; the `R<n>` code in the
  reason string ties each one back to a section above.

- The 3 collection errors under R3 have **no** mark — an import-time `TypeError`
  leaves pytest with no test item to attach a marker to. They must be fixed,
  not annotated.

- No files under `src/` were modified.

## Suggested order of attack

1. **R1** — `SSMFamily.RESPONSE_NDIM = 2`, drop the dead `create_extra_pps_coord`,
   and update the two hardcoded `rt,response_extra_dim_0` checks. Clears 253 of
   286 ids and unblocks real triage. Expect latent items 1-3 above to surface
   immediately afterwards.
2. **R3 + R4** — collapse `linkinv`/`linkinv_backend` into one PyTensor-compatible
   `inverse_link` in `src/hssm/link.py`. Clears 11 ids *and* the 3 collection
   errors. User-facing: `hssm.Link`'s signature and docstring change.
3. **R2** — teach `_make_truncated_dist`'s closure to accept and forward
   `dims`/`shape`.
4. **R5 + latent #1** — one job: replace `Model._compute_likelihood_params` and
   `Family._make_dist_kwargs_and_coords` with `pymc.compute_deterministics`
   against the conditional-parameter deterministics.
5. **R6 + latent #4** — update the `tests/test_utils.py` doubles and migrate the
   remaining deprecated accessors, resolving the `response_term` vs
   `response_component.term` inconsistency in `utils.py`.
6. **Latent #2 and #5** — audit `sample_posterior_predictive` against the new
   `predictions` group and the new new-group semantics. These are behaviour
   changes, so they need test coverage, not just a rename.
7. **R7 / R9** — decide intended behaviour, then update assertion or code.
8. **R8** — reproduce against the pre-upgrade environment to confirm it is the
   pytensor/numba bump, then report upstream if so.

Re-running after each step will reclassify the R1 collateral; expect the
inventory to churn considerably rather than shrink monotonically.
