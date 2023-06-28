# Likelihood functions in HSSM explained

One of the design goals of HSSM is its flexibility. It is built from ground up to support many types of likelihood functions out-of-the-box. For more sophisticated use cases, HSSM provides a convenient toolbox to help the user build their own likelihood functions that can be used with HSSM. This notebook focuses on explaining how to use different types of likelihoods with HSSM.

## 3 Kinds of Likelihoods

HSSM supports 3 kinds of likelihood functions supported via the `loglik_kind` parameter to the `HSSM` class:

- `"analytical"`: These likelihoods are usually closed-form solutions to the actual likelihoods. For example, For `ddm` models, HSSM provides the analytical likelihoods in Navarro & Fuss (2009). These likelihoods are typically Python functions written with `pytensor`, and thus can be compiled by `pytensor` as part of a computational graph. As such, they are differentiable as well.
- `"approx_differentiable"`: These likelihoods are usually approximations of the actual likelihood functions with neural networks. These networks can be trained with any popular deep learning framework such as `PyTorch` and `TensorFlow` and saved as `onnx` files. HSSM can load the `onnx` files and translate the information of the neural network with either the `jax` or the `pytensor` backends. Please see below for detailed explanations for these backends. The `backend` option can be supplied via the `"backend"` field via `model_config`. This field of `model_config` is not applicable to other kinds of likelihoods.

    - the `jax` backend: the feed-forward and back-propagation processes are computed with "JAX", which is wrapped in a `pytensor` `Op`. When sampling using the default NUTS sampler in `PyMC`, this option might be slightly faster but more prone to compatibility issues especially during parallel sampling due how `JAX` handles paralellism. When sampling using a JAX based sampler such as `nuts_numpyro` and `black_jax`, the `JAX` computation will be taken out of the `Op` and compiled together with the rest of the computation graph in `JAX`. Therefore, if a `JAX`-based sampler is used, this is a better option.
    - the `pytensor` backend: the feed-forward and back-propagation processes are computed with `pytensor`. When sampling using the default NUTS sampler in `PyMC`, this option allows for maximum compatibility. Not recommended when using `JAX`-based samplers.

- `"blackbox"`: Use this option for "black box" likelihoods that are not differentiable. These likelihoods are typically `Callable`s in Python that cannot be directly integrated to a `pytensor` computational graph. `hssm` will wrap these `Callable`s in a `pytensor` `Op` so it can be part of the graph.

## Default vs. Custom Likelihoods

HSSM provides many default likelihood functions out-of-the-box. The supported likelihoods are:

- For `analytical` kind: `ddm` and `ddm_sdv` models.
- For `approx_differentiable` kind: `ddm`, `ddm_sdv`, `angle`, `levy`, `ornstein`, `weibull`, `race_no_bias_angle_4` and `ddm_seq2_no_bias`.

For a model that has default likelihood functions, only `model` needs to be specified.

```python
ddm_model_analytical = hssm.HSSM(data, model="ddm")
```

`ddm` and `ddm_sdv` models have `analytical` and `approx_differentiable` likelihoods. If `loglik_kind` is not specified, the `analytical` likelihood will be used.

```python
ddm_model_approx_diff = hssm.HSSM(data, model="ddm", loglik_kind="approx_differentiable")
```

### Overriding default likelihoods

Sometimes a likelihood other than defaults is preferred. In that case, supply a likelihood function to the `loglik` parameter. We will discuss acceptable likelihood function types in a moment.

```python
ddm_model_analytical_override = hssm.HSSM(data, model="ddm", loglik=custom_logp_ddm)
```

## Using Custom Likelihoods

If you are specifying a model with a kind of likelihood that's not included in the list above, then HSSM considers that you are using a custom model with custom likelihoods. In this case, you will need to specify your entire model. Below is the procedure to specify a custom model:

1. Specify a `model` string. It can be any string that helps identify the model, but if it is not one of the model string supported in the `ssm_simulators` package [see full list here](https://github.com/AlexanderFengler/ssm-simulators/blob/main/ssms/config/config.py), you will need to supply a `RandomVariable` class to `model_config` detailed below. Otherwise, you can still perform MCMC sampling, but sampling from the posterior predictive distribution will raise a ValueError.

2. Specify a `model_config`. It typically contains the following fields:

    - `"list_params"`: Required if your `model` string is not one of `ddm`, `ddm_sdv`, `angle`, `levy`, `ornstein`, `weibull`, `race_no_bias_angle_4` and `ddm_seq2_no_bias`. A list of `str` indicating the parameters of the model.
    The order in which the parameters are specified in this list is important.
    Values for each parameter will be passed to the likelihood function in this
    order.
    - `"backend"`: Optional. Only used when `loglik_kind` is `approx_differentiable` and
    an onnx file is supplied for the likelihood approximation network (LAN).
    Valid values are `"jax"` or `"pytensor"`. It determines whether the LAN in
    ONNX should be converted to `"jax"` or `"pytensor"`. If not provided,
    `jax` will be used for maximum performance.
    - `"default_priors"`: Optional. A `dict` indicating the default priors for each parameter.
    - `"bounds"`: Optional. A `dict` of `(lower, upper)` tuples indicating the acceptable boundaries for each parameter. In the case
    of LAN, these bounds are training boundaries.
    - `"rv"`: Optional. Can be a `RandomVariable` class containing the user's own
    `rng_fn` function for sampling from the distribution that the user is
    supplying. If not supplied, HSSM will automatically generate a
    `RandomVariable` using the simulator identified by `model` from the
    `ssm_simulators` package. If `model` is not supported in `ssm_simulators`,
    a warning will be raised letting the user know that sampling from the
    `RandomVariable` will result in errors.

    **Note**
    `default_priors` and `bounds` in `model_config` specifies __default__ priors and bounds for the model. Actual priors and defaults should be provided via the `include` list and will override these defaults.

3. Specify `loglik` and `loglik_kind`.
4. Specify parameter priors in `include`.

Below are a few examples:

```python
# An angle model with an analytical likelihood function.
# Because `model` is known, no `list_params` needs to be provided.

custom_angle_model = hssm.HSSM(
    data,
    model="angle",
    model_config={
        "bounds": {
            "v": (-3.0, 3.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "t": (0.001, 2.0),
            "theta": (-0.1, 1.3),
        }  # bounds will be used to create Uniform (uninformative) priors by default
        # if priors are not supplied in `include`.
    },
    loglik=custom_angle_logp,
    loglik_kind="analytical",
)

# A fully customized model with a custom likelihood function.
# Because `model` is not known, a `list_params` needs to be provided.

my_custom_model = hssm.HSSM(
    data,
    model="my_model",
    model_config={
        "list_params": ["v", "a", "z", "t", "theta"],
        "bounds": {
            "v": (-3.0, 3.0),
            "a": (0.3, 3.0),
            "z": (0.1, 0.9),
            "t": (0.001, 2.0),
            "theta": (-0.1, 1.3),
        } # bounds will be used to create Uniform (uninformative) priors by default
          # if priors are not supplied in `include`.
        "default_priors": ... # usually no need to supply this.
        "rv": MyRV # provide a RandomVariable class if pps is needed.
    },
    loglik="my_model.onnx", # Can be a path to an onnx model.
    loglik_kind="approx_differentiable",
    include=[...]
)
```

## Supported likelihoods

When default likelihoods are not used, custom likelihoods are supplied via `loglik` argument to `HSSM`. Depending on what `loglik_kind` is used, `loglik` supports different types of Python objects:

- `Type[pm.Distribution]`: Supports all `loglik_kind`s.

    You can pass any **subclass** of `pm.Distribution` to `loglik` representing the underlying top-level distribution of the model. It has to be a class instead of an instance of the class.

- `Op`: Supports all `loglik_kind`s.

    You can pass a `pytensor` `Op` (an instance instead of the class itself), in which case HSSM will create a top-level `pm.Distirbution`, which calls this `Op` in its `logp` function to compute the log-likelihood.

- `Callable`: Supports all `loglik_kind`s.

    You can use any Python Callable as well. When `loglik_kind` is `blackbox`, HSSM will wrap it in a `pytensor` `Op` and create a top-level `pm.Distribution` with it. Otherwise, HSSM will assume that this Python callable is created with `pytensor` and is thus differentiable.

- `str` or `Pathlike`: Only supported when `loglik_kind` is `approx_differentiable`.

    The `str` or `Pathlike` indicates the path to the `onnx` file representing the neural network for likelihood approximation. In the case of `str`, if the path indicated by `str` is not found locally, HSSM will also look for the `onnx` file in the official HuggingFace repo. An error is thrown when the `onnx` file is not found.

**Note**

When using `Op` and `Callable` types of likelihoods, they need to have the this signature:

```
def logp_fn(data, *):
    ...
```

where `data` is a 2-column numpy array and `*` represents named arguments in the order of the parameters in `list_params`. For example, if a model's `list_params` is `["v", "a", "z", "t"]`, then the `Op` or `Callable` should at least look like this:

```
def logp_fn(data, v, a, z, t):
    ...
```

## Using `blackbox` likelihoods

HSSM also supports "black box" likelihood functions that are not differentiable. When `loglik_kind` is `blackbox`, by default, HSSM will switch to a MCMC sampler that does not use differentiation. Below is an example showing how to use a `blackbox` likelihood function. We choose a log-likelihood function for `ddm` written in Cython. [See here](https://github.com/brown-ccv/hddm-wfpt/blob/9107e4f1e480afcce2cd3cb7ac2279f8aecb596c/hddm_wfpt/wfpt.pyx#L32-L52) for the function definition.

```python
import hddm_wfpt

# Define a function with fun(data, *) signature
def pdf_ddm_blackbox(data, v, a, z, t, err=1e-4):
    # data is a 2 column array but the pdf array function accept a 1-d array
    # perform some pre-processing
    data = data[:, 0] * data[:, 1]

    # pdf_array is the Cython likelihood function for full DDM
    # from the original HDDM package.

    # Set some parameters to 0 to compute the likelihood for standard DDM.
    return hddm_wfpt.wfpt.pdf_array(data, v, 0, a, z, 0, t, 0, err, 1)

# Create the model with pdf_ddm_blackbox
model = hssm.HSSM(
    data=dataset,
    model="ddm",
    loglik=pdf_ddm_blackbox,
    loglik_kind="blackbox",
    model_config={
        "bounds": {
            "v": (-10.0, 10.0),
            "a": (2.0, 4.0),
            "z": (0.0, 1.0),
        }
    },
    # Specify the prior for `t`.
    t=bmb.Prior("Uniform", lower=0.0, upper=0.7, initval=0.1),
)

sample = model.sample()
```
