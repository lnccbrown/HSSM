# ONNX likelihood contract

This page is the canonical contract for an ONNX file used with
`loglik_kind="approx_differentiable"`, regardless of whether LANfactory, sbi,
BayesFlow, or another tool trained the network.

!!! info "Validation status"

    Documentation CI strictly builds this static contract. HSSM's loader and
    package tests enforce the concrete-dimension rule and smoke-load compliant
    artifacts; exporter parity and scientific recovery remain the artifact
    producer's responsibility.

## Required graph shape

An approximate differentiable ONNX likelihood represents exactly one trial:

- **Input:** one flat vector containing model parameters in `list_params` order,
  followed by the observed data columns. Its shape may be `(D,)` or `(1, D)`.
- **Output:** that trial's log-likelihood. It may be a scalar, `(1,)`, or
  `(1, 1)`, provided it squeezes to one value.
- **Dimensions:** every input dimension must be a concrete integer. Symbolic
  dimensions and `dynamic_axes` are forbidden.

HSSM batches the per-trial function itself with `jax.vmap`. Do not export a
dynamic or multi-trial batch axis for this route. A concrete singleton leading
dimension such as `(1, D)` remains valid.

## Why dynamic dimensions are rejected

`jaxonnxruntime` traces an ONNX graph against its construction-time input shape
and can bake those shapes into the translated closure. A symbolic batch axis can
therefore produce numerically wrong values at another batch size without a
clear runtime failure, especially when a graph contains a batch-dependent
`Reshape` or a flow log-determinant accumulator.

Single-trial export followed by HSSM-side vectorization is mathematically
equivalent for a per-trial likelihood and removes that silent-corruption path.
HSSM rejects symbolic input dimensions when it loads the graph.

## Rank is exporter-specific

Rank is not the invariant; concrete dimensions are. Supported ecosystem
exporters legitimately produce both forms:

| Exporter | Traced input | Typical lowering |
| --- | --- | --- |
| LANfactory Torch LAN/CPN/OPN | `(1, D)` | `Gemm` |
| LANfactory JAX LAN/CPN/OPN | `(1, D)` | `Gemm` |
| LANfactory sbi | `(D,)` | `MatMul` + `Add` |
| LANfactory BayesFlow | `(D,)` | `MatMul` + `Add` |

Flow graphs that slice a combined parameter/observation vector must use a
rank-1 dummy. A `(1, D)` flow trace can emit `Slice` operations on axis 1 that
fail after HSSM vectorizes the function. Plain feed-forward LANs work at either
rank. Match the exporter and rely on its contract assertion rather than copying
another exporter's dummy shape.

## Precision constraint for flow graphs

Flow-based exports can contain the `INT64_MAX` sentinel used for open-ended
slices. With `hssm.set_floatX("float32")`, truncating that constant would change
the graph. HSSM raises a `ValueError` instead. Use HSSM's default float64 setting
for flow-based ONNX likelihoods.

## Input ordering

HSSM supplies values in this order:

1. model parameters in `list_params` order; then
2. the observed data columns, normally reaction time and response.

The exporter and `ModelConfig` must agree on that order. A dimensionally valid
graph with a different column order can still return plausible but incorrect
likelihoods.

## Producer verification checklist

Before publishing an artifact:

1. inspect the ONNX input and confirm that every dimension is concrete;
2. compare the source model and ONNX Runtime across in-bounds parameter draws;
3. use an exporter tolerance appropriate to the model (the ecosystem exporters
   use `atol=1e-4` as the outer parity bound);
4. smoke-load the file in a real `hssm.HSSM` model and require a finite initial
   log-probability; and
5. run parameter recovery before using the likelihood for scientific claims.

LANfactory exporters provide
`lanfactory.onnx.contract.assert_single_trial_contract` for the first check,
plus ONNX checker, runtime-session, input-width, and optional operator checks.
It does not construct an HSSM model or evaluate its log-probability. The
exporter documentation owns framework-specific training constraints; HSSM owns
this consumer contract and its model-level smoke test.

## Black-box ONNX is a separate route

The [black-box ONNX walkthrough](../tutorials/blackbox_contribution_onnx_example.ipynb)
uses ONNX Runtime inside an ordinary Python function and may rewrite a graph to
accept dynamic batches. HSSM never translates that graph to JAX. That procedure
applies only to `loglik_kind="blackbox"` and is deliberately incompatible with
the approximate differentiable contract above.

See [Likelihood kinds in HSSM](../explanations/likelihoods.md) for the choice
between routes and [Bring your own likelihood](external_trainers.md) for the
supported external-trainer paths.
