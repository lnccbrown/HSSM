# Issue #1305 — bambi 0.20 migration: fix plan

Companion to [`bambi-dev-upgrade-test-failures.md`](./bambi-dev-upgrade-test-failures.md),
which inventories *what* fails and *why*. This document is the *what to do
about it* list: nine work items, each sized to one PR.

**Upstream change:** [bambinos/bambi#1002 "Bambi rewrite"](https://github.com/bambinos/bambi/pull/1002)
(merged 2026-09-07) — a frontend/backend split, not a feature release.
**Tracking parent:** #1306.

## Current state

```
673 passed, 5 skipped, 286 xfailed, 0 failed, 0 xpassed
+ 3 collection errors that cannot be xfail-marked
```

286 failing test ids across 143 functions in 33 files are marked
`@pytest.mark.xfail(reason="bambi 0.20 migration (#1305): R<n> …", strict=False)`.
Marks are applied per *function*, so the 145 marks cover 286 ids.

## Fix items

All nine are tracked as sub-issues of #1306.

| # | Issue | Fix | Root causes | Unblocks | Risk |
|---|-------|-----|-------------|---------:|------|
| F1 | #1310 | Declare `SSMFamily.RESPONSE_NDIM = 2`, drop dead hook, fix coord name | R1 | 253 ids | Low |
| F2 | #1311 | Collapse `hssm.Link` onto `inverse_link` | R3, R4 | 11 ids + 3 collection errors | Medium (user-facing) |
| F3 | #1312 | Let truncated-prior callables accept `dims` | R2 | 9 ids | Low |
| F4 | #1313 | Replace removed likelihood-parameter APIs | R5 + latent #1 | 6 ids | Medium |
| F5 | #1314 | Migrate deprecated accessors, refresh test doubles | R6 + latent #4 | 3 ids | Low |
| F6 | #1315 | Handle the new `predictions` DataTree group | latent #2 | 0 (masked) | High (behavioral) |
| F7 | #1316 | Adapt to new-group prediction semantics | latent #5 | 0 (masked) | High (behavioral) |
| F8 | #1317 | Reconcile two drifted assertions | R7, R9 | 2 ids | Low |
| F9 | #1318 | Investigate numba slice-sampler `SystemError` | R8 | 2 ids | Unknown (likely upstream) |

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

- [ ] `src/hssm/distribution_utils/dist.py:753` — set `RESPONSE_NDIM = 2`
- [ ] `src/hssm/distribution_utils/dist.py:756` — remove `create_extra_pps_coord`
- [ ] `src/hssm/base.py:1287`, `src/hssm/base.py:1356` — the hardcoded
      `"rt,response_extra_dim_0"` coord is now `rt_dim`; these checks currently
      never match
- [ ] Re-run the suite and re-triage; remove the R1 xfail marks that now pass

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

- [ ] `src/hssm/prior.py:168-180` — accept and forward `dims` (and `shape`)
- [ ] Confirm behaviour for both bounded and unbounded priors, and for
      regression intercepts specifically
- [ ] Remove the R2 xfail marks

### F4 (#1313) — Replace the removed likelihood-parameter APIs

**Root cause:** R5 (6 ids) + latent #1

Two deletions, one job. PR #1002 removes `Model._compute_likelihood_params` and
lists `Family._make_dist_kwargs_and_coords` by name: "distribution arguments and
coordinates are no longer reconstructed from posterior samples."

The PR names the replacement under *Graph evaluation and interventions*:
"Conditional parameters are represented as deterministic variables in the PyMC
model. The backend can evaluate them with `pymc.compute_deterministics`."

- [ ] `src/hssm/base.py:1028`, `src/hssm/base.py:1034` —
      `_compute_likelihood_params`
- [ ] `src/hssm/utils.py:205` — `_compute_likelihood_params`
- [ ] `src/hssm/utils.py:286` — `_make_dist_kwargs_and_coords` (already flagged
      by `pyrefly`)
- [ ] Remove the R5 xfail mark on `test_choice_only` — note it also covers the
      2 R8 ids, which will still fail (see F9)

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

- [ ] `src/hssm/utils.py:134`, `src/hssm/base.py:1812` — `components`
- [ ] `src/hssm/utils.py:127`, `:142`, `src/hssm/base.py:915` —
      `distributional_components`
- [ ] `src/hssm/utils.py:280`, `:285` — `response_component`
- [ ] `tests/test_utils.py` — the `SimpleNamespace` doubles still mimic the
      pre-rewrite API
- [ ] Consider failing CI on bambi `FutureWarning`s to catch the rest
- [ ] Remove the R6 xfail marks

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

- [ ] Audit `src/hssm/base.py:1039-1214` for the group assumption
- [ ] Decide whether HSSM surfaces `predictions` or normalises it back into
      `posterior_predictive` for API stability
- [ ] Add out-of-sample prediction tests

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

- [ ] Decide intended behaviour for each, then update assertion or code
- [ ] Remove the R7/R9 xfail marks

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

---

## Definition of done for the migration

- [ ] No `xfail` marks referencing `bambi 0.20 migration (#1305)` remain
- [ ] No collection errors
- [ ] No bambi `FutureWarning`s emitted from `src/hssm/`
- [ ] `uv run prek run --all-files` clean (`ruff`, `pyrefly`, `mypy`)
- [ ] Notebook CI green (`check_notebooks.yml`) — not covered by this inventory
- [ ] Breaking changes recorded in `docs/changelog.md`, `hssm.Link` in particular
