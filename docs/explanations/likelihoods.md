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

## Declaring where the response-time support starts

HSSM floors the log-likelihood of every response time at or below the non-decision time `t`, because a sequential sampling model assigns no density there. Some likelihoods admit responses *below* `t`: the `hddm_wfpt` likelihood behind `full_ddm` reads `st` as the full width of the non-decision-time distribution, so its support starts at `t - st / 2`. Such a likelihood declares where its support starts with an `ndt_edge_shift` entry, `{"param": "<name>", "scale": s}` (the `hssm._types.NDTEdgeShift` `TypedDict`), which moves the floor to `t - s * <name>`. The declaration can be given in three places:

- in the likelihood dict of a model registered with `hssm.register_model`, so it ships with the likelihood;
- as an `"ndt_edge_shift"` field of `model_config` when constructing `hssm.HSSM(...)`, which overrides the registered declaration;
- as the `ndt_edge_shift=` keyword of `make_distribution` when building a `pm.Distribution` directly (see [Use the low-level API with PyMC](https://lnccbrown.github.io/HSSM/tutorials/pymc/)).

`full_ddm` declares `{"param": "st", "scale": 0.5}`. A likelihood without the key keeps the floor at `t`, and a model without `t` in its `list_params` has no floor at all. For a network trained on `ssm-simulators` output, the declaration must match the simulator's: the network does not learn the edge, its training labels merely smooth across it, and the floor is what restores the model's zero density below the edge. For example, a registered model whose non-decision time varies uniformly between `t - st` and `t + st` declares the edge at `t - st`:

```python
hssm.register_model(
    name="my_ddm_st",
    response=["rt", "response"],
    list_params=["v", "a", "z", "t", "st"],
    choices=[-1, 1],
    description="A DDM whose non-decision time is Uniform(t - st, t + st)",
    likelihoods={
        "blackbox": {
            "loglik": my_ddm_st_logp,
            "backend": None,
            "default_priors": {},
            "bounds": {
                "v": (-3.0, 3.0),
                "a": (0.3, 3.0),
                "z": (0.1, 0.9),
                "t": (0.001, 2.0),
                "st": (0.0, 0.5),
            },
            "extra_fields": None,
            # Responses are admissible from t - st onwards, not from t.
            "ndt_edge_shift": {"param": "st", "scale": 1.0},
        }
    },
)
```
