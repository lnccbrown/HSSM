# Commit 2 — attention-process registry + likelihood Op builder

Adds the two pure-HSSM pieces that turn the Commit 1 vendored JAX kernel into a
usable, differentiable likelihood. Mirrors `src/hssm/rl/likelihoods/builder.py`
and reuses HSSM's existing JAX→PyTensor machinery (no new Op infrastructure).

## What was added

| File | Purpose |
|---|---|
| `src/hssm/addm/attention_process.py` | Name→callable registry. `standard_alternating` (the default) delegates to the vendored `_build_addm_mu_array_data`, so the default attention process and the default kernel agree by construction. `resolve_attention_process(name_or_callable)` resolves a registry key or callable. |
| `src/hssm/addm/likelihoods/builder.py` | `make_addm_logp_func(attention_process)` → JAX `logp(data, eta, kappa, a, b, x0, r1, r2, flag, sacc_array, d, sigma)`; `make_addm_logp_op(attention_process, list_params, extra_fields)` → differentiable PyTensor `Op`. |
| `src/hssm/addm/likelihoods/__init__.py` | Now exports `make_addm_logp_func`, `make_addm_logp_op`. |

## Design notes

- **Column-ordering contract** (verified against RLSSM): the distribution calls
  `loglik(data, *dist_params, *extra_fields)`. With
  `list_params = [eta, kappa, a, b, x0]` and
  `extra_fields = [r1, r2, flag, sacc_array, d, sigma]`, the closure reorders
  these into the kernel's positional slots (note `sigma` is the last extra field
  but the kernel's 5th argument).
- **Default vs custom path**: default `standard_alternating` calls
  `compute_addm_loglikelihoods` (kernel builds the drift array internally); any
  other process produces `mu` and calls `compute_addm_loglikelihoods_from_mu`
  (the Commit 1 wrapper). Branch is decided once at build time, jit-friendly.
- **Gradients**: `make_addm_logp_op` reuses `make_vjp_func` and
  `make_jax_logp_ops` from `hssm.distribution_utils`. `n_params = len(list_params)`
  → gradients only for the sampled params; extra-field gradients are undefined.
- **Import-cycle fix (only deviation):** `attention_process.standard_alternating`
  imports the vendored kernel **lazily** (inside the function). A top-level import
  there would trigger `likelihoods/__init__` → `builder` → `attention_process`
  before `resolve_attention_process` is defined. The lazy import breaks the cycle
  and the kernel is only needed when drift is actually computed. Faithful
  top-level imports are kept everywhere else (including the heavy pytensor
  imports in `builder.py`).

## Tests — `test_commit2_builder.py`

Run in the repo's uv venv (`source .venv/bin/activate`), which resolves `hssm` to
this repo's `src` and carries pytensor/pymc/bambi:

```
JAX_PLATFORMS=cpu python src/hssm/addm/commit_tests/test_commit2_builder.py
```

1. `test_logp_func_output_shape` — `(n_trials,)`, dtype `get_jax_dtype()`.
2. `test_logp_matches_kernel` — logp == direct `compute_addm_loglikelihoods` (1e-6).
3. `test_logp_op_matches_func` — PyTensor `Op.eval()` == raw JAX logp (1e-6).
4. `test_gradients_finite` — `jax.grad` finite w.r.t. eta, kappa, a, b, x0, sigma.
5. `test_custom_attention_process` — identity-scaled custom process matches the
   default (confirms `from_mu` routing); 2×-scaled drift differs (confirms the
   custom branch is taken).

**Status:** all 5 pass under the repo venv (jax 0.4.x, x64).
