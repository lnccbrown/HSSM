# Likelihood kinds in HSSM

Every HSSM model needs a function that scores the observed response and reaction
time under a set of model parameters. HSSM calls that function a likelihood and
supports three kinds. The distinction matters because it determines which
samplers are available, where the numerical implementation comes from, and how
a custom model must be configured.

## Analytical likelihoods

An `analytical` likelihood is a differentiable numerical implementation owned
by HSSM. It may use PyTensor operations directly or provide a JAX callable that
HSSM wraps for use inside a PyTensor graph. Gradient-based PyMC samplers can use
either backend. The label describes the implementation route; some functions
use accurate numerical approximations rather than a single closed-form
expression.

HSSM uses the analytical route by default whenever a built-in model provides
one. This includes the DDM and DDM-SDV, the LBA models, racing diffusion,
Poisson race, and the softmax choice models. See the
[built-in model and likelihood matrix](../reference/models-and-likelihoods.md)
for the current list.

## Approximate differentiable likelihoods

An `approx_differentiable` likelihood is a learned or otherwise approximate
function that HSSM can differentiate. The usual artifact is a single-trial ONNX
network translated to JAX, although a compatible JAX callable can also provide
the likelihood. HSSM vectorizes the single-trial function across observations.

This route makes models without an analytical likelihood available to
gradient-based samplers. Its validity depends on the training domain and on
simulation-based validation: differentiability does not guarantee that a
network is accurate outside the parameter region it learned.

ONNX artifacts must satisfy the exact
[ONNX likelihood contract](../how_to/custom_onnx_likelihoods.md). To choose an
external training route, use [Bring your own likelihood](../how_to/external_trainers.md).

## Black-box likelihoods

A `blackbox` likelihood is an ordinary Python, PyTensor, or ONNX-backed function
for which HSSM cannot provide gradients. It is the most flexible route, but it
requires a sampler that does not depend on likelihood gradients. The black-box
ONNX walkthrough deliberately permits a batched dynamic graph; that is a
different execution path from the concrete single-trial graph required by the
approximate differentiable route.

Use [Custom models from ONNX files](../tutorials/blackbox_contribution_onnx_example.ipynb)
for the black-box procedure. Do not apply its dynamic-axis rewrite to an
`approx_differentiable` artifact.

## Defaults and overrides

When `loglik_kind` is omitted for a built-in `hssm.HSSM` model, HSSM selects the
first available kind in this order:

1. `analytical`;
2. `approx_differentiable`; and
3. `blackbox`.

Passing `loglik_kind` requests a specific configured route. Passing `loglik`
overrides the corresponding default function or artifact. A custom model also
needs the response columns, parameter order, choices, and likelihood metadata
described by [`hssm.ModelConfig`](../api/model_config.md) or
[`hssm.register_model`](../api/model_registry.md).

The [built-in model and likelihood matrix](../reference/models-and-likelihoods.md)
is the canonical catalog. Exact constructor rules live in the
[`hssm.HSSM` API reference](../api/hssm.md).
