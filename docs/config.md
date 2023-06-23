# Module: Default Configurations for HSSM Models

This module provides default configurations for various models used in the Hierarchical Sequential Sampling Models (HSSM) class.

## Model Configurations:

The module includes a dictionary, `default_model_config`, that provides default configurations for a variety of models, including:

- **DDM**
- **Angle**
- **Levy**
- **Ornstein**
- **Weibull**
- **Race_no_bias_angle_4**
- **DDM_seq2_no_bias**

The configuration for each model includes parameters like the type of log-likelihood (`loglik`), kind of log-likelihood (`loglik_kind`), list of parameters specific to the model (`list_params`), the computational backend to be used (`backend`), and the bounds for the parameters (`bounds`).

### Model: DDM

- **Log-likelihood:** wfpt.WFPT
- **Log-likelihood kind:** Analytical
- **Parameters:** v, sv, a, z, t
- **Backend:** pytensor
- **Bounds:**
  - v: (-3.0, 3.0)
  - sv: (0.0, 1.0)
  - a: (0.3, 2.5)
  - z: (0.1, 0.9)
  - t: (0.0, 2.0)

---

### Model: Angle

- **Log-likelihood kind:** Approximate differentiable
- **Log-likelihood:** angle.onnx
- **Parameters:** v, a, z, t, theta
- **Backend:** jax
- **Bounds:**
  - v: (-3.0, 3.0)
  - a: (0.3, 3.0)
  - z: (0.1, 0.9)
  - t: (0.001, 2.0)
  - theta: (-0.1, 1.3)

---

### Model: Levy

- **Log-likelihood kind:** Approximate differentiable
- **Log-likelihood:** levy.onnx
- **Parameters:** v, a, z, alpha, t
- **Backend:** jax
- **Bounds:**
  - v: (-3.0, 3.0)
  - a: (0.3, 3.0)
  - z: (0.1, 0.9)
  - alpha: (1.0, 2.0)
  - t: (1e-3, 2.0)

---

### Model: Ornstein

- **Log-likelihood kind:** Approximate differentiable
- **Log-likelihood:** ornstein.onnx
- **Parameters:** v, a, z, g, t
- **Backend:** jax
- **Bounds:**
  - v: (-2.0, 2.0)
  - a: (0.3, 3.0)
  - z: (0.1, 0.9)
  - g: (-1.0, 1.0)
  - t: (1e-3, 2.0)

---

### Model: Weibull

- **Log-likelihood kind:** Approximate differentiable
- **Log-likelihood:** weibull.onnx
- **Parameters:** v, a, z, t, alpha, beta
- **Backend:** jax
- **Bounds:**
  - v: (-2.5, 2.5)
  - a: (0.3, 2.5)
  - z: (0.2, 0.8)
  - t: (1e-3, 2.0)
  - alpha: (0.31, 4.99)
  - beta: (0.31, 6.99)

---

### Model: Race_no_bias_angle_4

- **Log-likelihood kind:** Approximate differentiable
- **Log-likelihood:** race_no_bias_angle_4.onnx
- **Parameters:** v0, v1, v2, v3, a, z, ndt, theta
- **Backend:** jax
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

- **Log-likelihood kind:** Approximate differentiable
- **Log-likelihood:** ddm_seq2_no_bias.onnx
- **Parameters:** vh, vl1, vl2, a, t
- **Backend:** jax
- **Bounds:**
  - vh: (-4.0, 4.0)
  - vl1: (-4.0, 4.0)
  - vl2: (-4.0, 4.0)
  - a: (0.3, 2.5)
  - t: (0.0, 2.0)