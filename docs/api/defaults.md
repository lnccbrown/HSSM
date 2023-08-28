# Module: Default Configurations for HSSM Models

This module provides default configurations for various models used in the Hierarchical Sequential Sampling Models (HSSM) class.

## Model Configurations:

The module includes a dictionary, `default_model_config`, that provides default configurations for a variety of models, including:

- `ddm`
- `ddm_sdv`
- `full_ddm`
- `angle`
- `levy`
- `ornstein`
- `weibull`
- `race_no_bias_angle_4`
- `ddm_seq2_no_bias`

## Configuration parameters

Each model configuration is specified by several parameters, which include:

- `loglik`
- `list_params`
- `default_priors`
- `backend`
- `bounds`

## Default Configurations

For each model, a dictionary is defined containing configurations for each `LoglikKind`. Each configuration includes:

- `loglik`: the log-likelihood function or filename
- `bounds`: the bounds for the model parameters
- `default_priors`: the default priors for the model parameters
- `backend`: (optional) the backend for approximating the likelihood

### Model: DDM

#### Analytical
- **Log-likelihood kind:** Analytical
- **Log-likelihood:** log_pdf
- **Parameters:** v, a, z, t
- **Bounds:**
  - z: (0.0, 1.0)
- **Default priors:**
  - v: Uniform (-10.0, 10.0)
  - a: HalfNormal with sigma 2.0
  - t: Uniform (0.0, 0.5) with initial value 0.1

#### Approx Differentiable
- **Log-likelihood kind:** Approx Differentiable
- **Log-likelihood:** ddm.onnx
- **Backend:** jax
- **Parameters:** v, a, z, t
- **Bounds:**
  - v: (-3.0, 3.0)
  - a: (0.3, 2.5)
  - z: (0.1, 0.9)
  - t: (0.0, 2.0)

---

### Model: DDM_SDV

#### Analytical
- **Log-likelihood kind:** Analytical
- **Log-likelihood:** log_pdf_sv
- **Parameters:** v, sv, a, z, t
- **Bounds:**
  - z: (0.0, 1.0)
- **Default priors:**
  - v: Uniform (-10.0, 10.0)
  - sv: HalfNormal with sigma 2.0
  - a: HalfNormal with sigma 2.0
  - t: Uniform (0.0, 5.0) with initial value 0.1

#### Approx Differentiable
- **Log-likelihood kind:** Approx Differentiable
- **Log-likelihood:** ddm_sv.onnx
- **Backend:** jax
- **Parameters:** v, sv, a, z, t
- **Bounds:**
  - v: (-3.0, 3.0)
  - sv: (0.0, 1.0)
  - a: (0.3, 2.5)
  - z: (0.1, 0.9)
  - t: (0.0, 2.0)

---

### Model: Ornstein

- **Log-likelihood kind:** Approx Differentiable
- **Log-likelihood:** ornstein.onnx
- **Backend:** jax
- **Parameters:** v, a, z, g, t
- **Bounds:**
  - v: (-2.0, 2.0)
  - a: (0.3, 3.0)
  - z: (0.1, 0.9)
  - g: (-1.0, 1.0)
  - t: (1e-3, 2.0)

---

### Model: Weibull

- **Log-likelihood kind:** Approx Differentiable
- **Log-likelihood:** weibull.onnx
- **Backend:** jax
- **Parameters:** v, a, z, t, alpha, beta
- **Bounds:**
  - v: (-2.5, 2.5)
  - a: (0.3, 2.5)
  - z: (0.2, 0.8)
  - t: (1e-3, 2.0)
  - alpha: (0.31, 4.99)
  - beta: (0.31, 6.99)

---

### Model: Race_no_bias_angle_4

- **Log-likelihood kind:** Approx Differentiable
- **Log-likelihood:** race_no_bias_angle_4.onnx
- **Backend:** jax
- **Parameters:** v0, v1, v2, v3, a, z, ndt, theta
- **Bounds:**
  - v0: (0.0, 2.5)
  - v1: (0.0, 2.5)
  - v2: (0.0, 2.5)
  - v3: (0.0, 2.5)
  - a: (1.0, 3.0)
  - z: (0.0, 0.9)
  - ndt: (0.0, 2.0)
  - theta: (-0.1, 1.45)

---

### Model: DDM_seq2_no_bias

- **Log-likelihood kind:** Approx Differentiable
- **Log-likelihood:** ddm_seq2_no_bias.onnx
- **Backend:** jax
- **Parameters:** vh, vl1, vl2, a, t
- **Bounds:**
  - vh: (-4.0, 4.0)
  - vl1: (-4.0, 4.0)
  - vl2: (-4.0, 4.0)
  - a: (0.3, 2.5)
  - t: (0.0, 2.0)

## WFPT and WFPT_SDV Classes

The WFPT and WFPT_SDV classes are created using the make_distribution function. They represent the Drift Diffusion Model (`ddm`) and Drift Diffusion Model with inter-trial variability in drift (`ddm_sdv`) respectively. They use the log-likelihood functions and parameter lists from the default configurations and parameters.
