# Vendored JAX aDDM likelihood — provenance & license

This directory is a frozen, in-tree copy of the pure-JAX aDDM log-likelihood from
the **efficient-fpt** project. It is *vendored* (not depended on) so HSSM gains the
differentiable kernel without efficient-fpt's heavyweight Cython build chain or a
new runtime dependency — JAX/jaxlib are already HSSM dependencies.

> **Do not edit these files in place.** To update, re-vendor the whole directory
> from upstream and re-apply the HSSM-authored modifications listed below.

## License & attribution

The vendored code is © 2025 Sicheng Liu, released under the MIT License — see the
[`LICENSE`](LICENSE) file in this directory, reproduced verbatim as MIT requires
(HSSM's own top-level license differs). efficient-fpt is an in-house ecosystem
project; the intended end-state is to **absorb & relicense** this code under
HSSM's license.

- **TODO (relicense):** obtain the author's sign-off to relicense under HSSM's
  license; until then this MIT notice governs the vendored subtree.

## Upstream

| | |
|---|---|
| Project | efficient-fpt |
| Commit | `d97a451479141acef845195610f0f9d85824844e` |
| Source | `src/efficient_fpt/jax/` (+ two support modules from `src/efficient_fpt/`) |

## Vendored files

| Vendored file | Upstream source |
|---|---|
| `batch.py` | `src/efficient_fpt/jax/batch.py` |
| `multi_stage.py` | `src/efficient_fpt/jax/multi_stage.py` |
| `single_stage.py` | `src/efficient_fpt/jax/single_stage.py` |
| `addm_helpers.py` | `src/efficient_fpt/jax/addm_helpers.py` |
| `utils.py` | `src/efficient_fpt/jax/utils.py` |
| `_defaults.py` | `src/efficient_fpt/_defaults.py` |
| `_quadrature.py` | `src/efficient_fpt/quadrature.py` |

## HSSM-authored modifications (re-apply on re-vendor)

1. **Import retargeting** so the subpackage is self-contained (re-vendor = one
   directory):
   - `batch.py`, `multi_stage.py`:
     - `from .._defaults import ...` → `from ._defaults import ...`
     - `from ..utils import resolve_quadrature_orders` → `from .utils import resolve_quadrature_orders`
   - `utils.py`:
     - `from ..quadrature import ...` → `from ._quadrature import ...`
2. **`utils.py`**: `resolve_quadrature_orders` folded in from
   `src/efficient_fpt/utils.py` (only that function; `adaptive_interpolation` was
   not copied).
3. **`batch.py`**: a public `compute_addm_loglikelihoods_from_mu` wrapper added
   over the private `_compute_addm_loglikelihoods_batchscan_core`, giving a stable
   entry that accepts a pre-built per-trial drift array.
4. **`_defaults.py`**: `DEFAULT_TRUNC_NUM = 6` (efpt upstream is `100`). This is a
   deliberate, RAM-driven change for the JAX path: the kernel unrolls the FPT
   truncation series into the compiled graph and its reverse-mode gradient (VJP)
   materializes per-term intermediates, so `trunc=100` can exhaust RAM at
   compile/grad time. `6` keeps the graph tractable at a measured ~6e-4 relative
   FPT-density error versus the efpt NumPy oracle (`tests/addm/oracle/`, which
   pins this kernel to ~1e-6 under x64) — acceptable for inference. **Keep this
   `<= 6` on the JAX path.**

## Not vendored

`efficient_fpt/cython/`, `efficient_fpt/numpy/`, `efficient_fpt/models.py`, the
simulator, and the remainder of `efficient_fpt/quadrature.py` / `utils.py` — none
are on the HSSM inference path. (efpt's NumPy backend is vendored separately, and
only as a **test** oracle, under `tests/addm/oracle/`.)
