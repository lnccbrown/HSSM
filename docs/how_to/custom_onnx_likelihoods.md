# The ONNX likelihood contract

This page is the canonical statement of the rules an ONNX file must follow to
be used as an HSSM likelihood via `loglik_kind="approx_differentiable"` —
whether it was trained with [LANfactory](https://github.com/lnccbrown/LANfactory),
BayesFlow, sbi, or anything else that can emit ONNX.

!!! tip "The contract in two sentences"

    Export **one per-trial forward pass** with **every input dimension
    concrete** — no `dynamic_axes`, no symbolic shapes. HSSM batches across
    trials itself, by wrapping your graph in `jax.vmap`.

## What HSSM expects of the graph

- **Input:** a single flat per-trial vector containing the model parameters
  first, then the data columns — shape `(n_params + n_data_cols,)` or
  `(1, n_params + n_data_cols)` (see [Rank](#rank-what-is-and-is-not-required)
  below). For a DDM-family LAN that means `(v, a, z, t, rt, choice)`.
- **Output:** the log-likelihood of that single trial — a scalar, or any shape
  that squeezes to one (`()`, `(1,)`, `(1, 1)`).
- **Every dimension concrete.** If any input dimension is symbolic or dynamic,
  HSSM refuses to load the file (see
  [Errors you will see](#errors-you-will-see)).

## Why: the silent-corruption failure mode

HSSM converts ONNX to JAX with `jaxonnxruntime`, which traces your graph
against its construction-time input shape and **bakes the resulting shapes
into the returned closure**. A graph exported with a dynamic batch axis does
not fail loudly when called at a different batch size — it silently returns
**wrong numbers** for any model with a batch-dependent intermediate (a log-det
accumulator in a normalizing flow, a `Reshape` whose `-1` resolves against the
batch dimension).

Single-trial export plus HSSM-side `jax.vmap` is mathematically equivalent —
the likelihood is per-trial — has zero JIT overhead after XLA fusion, and
makes the failure mode impossible. This is why the contract is enforced at
load time instead of documented as a recommendation.

## Rank: what is and is not required

The invariant is *concrete dimensions*. **Rank is not part of the contract** —
it follows from how your tracer lowers a dense layer, and the ecosystem
legitimately contains both forms:

| exporter | traced dummy | lowering |
|----------|--------------|----------|
| LANfactory `transform_onnx.py` (LAN/CPN/OPN, torch) | `(1, D)` | `Gemm` |
| LANfactory `jax_export.py` (LAN/CPN/OPN, jax2onnx) | `(1, D)` | `Gemm` |
| LANfactory `sbi.py` | `(D,)` | `MatMul` + `Add` |
| LANfactory `bayesflow.py` | `(D,)` | `MatMul` + `Add` |

`torch.onnx.export` lowers `Linear` to rank-agnostic `MatMul`+`Add`, so a
rank-1 dummy works. `jax2onnx` lowers flax `Dense` to `Gemm`, whose ONNX spec
requires rank 2. Both forms load in HSSM and, measured under `vmap`+`jit`, run
identically. All production networks on
[`franklab/HSSM`](https://huggingface.co/franklab/HSSM) are `(1, D)` `Gemm`.

!!! warning "One real rank constraint: flow-based graphs"

    Graphs that internally *slice* their input (sbi and BayesFlow flow
    exports split the input into `theta` and `x`) must be traced with a
    **rank-1** dummy: a `(1, D)` trace emits `Slice` ops with `axes=[1]`
    that fail under HSSM's `vmap`. Pure feed-forward MLPs (LANs) are fine at
    either rank. When in doubt, trace rank-1 if `torch.onnx.export` is your
    tracer.

## Errors you will see

- **`ValueError: ... dynamic/symbolic input dimension ...`** at model build —
  your graph has a non-concrete input dim. Re-export without `dynamic_axes`.
- **`ValueError: ... int64 constant outside the int32 range ...`** — you are
  running `hssm.set_floatX("float32")` with a flow-based export. Flow graphs
  carry an `INT64_MAX` open-ended-slice sentinel that JAX truncates to `-1`
  when x64 is off, which would corrupt the likelihood; HSSM raises instead.
  Use the default `float64` setting for flow-based ONNX likelihoods.

## Exporting correctly

With raw `torch.onnx.export`: pass a single-trial dummy and **omit
`dynamic_axes` entirely**:

```python
torch.onnx.export(
    network,
    torch.zeros(n_params + n_data_cols),  # rank-1 single-trial dummy
    "my_model.onnx",
    # no dynamic_axes
)
```

If you trained with LANfactory, BayesFlow, or sbi, prefer LANfactory's
exporters (`transform_onnx`, `transform-jax-onnx`, `transform_sbi_to_onnx`,
and the BayesFlow export) — they follow the contract by construction. Whatever
the route, add LANfactory's executable check to your tests instead of
re-deriving the rules:

```python
from lanfactory.onnx.contract import assert_single_trial_contract

assert_single_trial_contract("my_model.onnx", expected_input_width=6)
```

## Not to be confused with the blackbox route

The tutorial
[Custom models from ONNX files (blackbox)](../tutorials/blackbox_contribution_onnx_example.ipynb)
shows a trick that rewrites input dimensions to dynamic to enable batched
`onnxruntime` inference. That applies **only** to `loglik_kind="blackbox"`,
where the ONNX file is executed by `onnxruntime` inside an ordinary Python
function and HSSM never converts the graph. The very same rewritten file will
be **rejected** by the `approx_differentiable` loader described on this page.
The two routes are opposites on this point; pick the route first, then shape
the file.

## Verification checklist

Before publishing or sharing an ONNX likelihood:

1. `assert_single_trial_contract(path)` passes.
2. Round-trip parity: `onnxruntime` output matches your source model on ~1000
   in-bounds parameter draws (`atol=1e-4`).
3. HSSM smoke-load: build an `hssm.HSSM` model on a small simulated dataset
   with `loglik=path`, `loglik_kind="approx_differentiable"`; the initial
   log-probability is finite.
4. The input column order matches your `list_params` + data columns — HSSM
   passes parameters in `list_params` order, then `(rt, response)`.

## See also

- [Likelihood functions in HSSM explained](../tutorials/likelihoods.ipynb) —
  the analytical / approx_differentiable / blackbox taxonomy
- [Using HSSM low-level API directly with PyMC](../tutorials/pymc.ipynb) —
  `make_likelihood_callable` and friends
- [Custom models from JAX callables](../tutorials/jax_callable_contribution_onnx_example.ipynb)
  — the same idea without ONNX, via a JAX log-likelihood function
