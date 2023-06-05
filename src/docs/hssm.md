# HSSM - Hierarchical Sequential Sampling Modeling

## Class: HSSM

### Introduction

The Hierarchical Sequential Sampling Model (HSSM) class.

### Attributes

- `data`: A pandas DataFrame with at least two columns of "rt" and "response" indicating the response time and responses.
- `model`: The name of the model to use. Currently supported models are "ddm", "angle", "levy", "ornstein", "weibull", "race_no_bias_angle_4", "ddm_seq2_no_bias". If using a custom model, please pass "custom". Defaults to "ddm".
- `include`: A list of dictionaries specifying parameter specifications to include in the model. If left unspecified, defaults will be used for all parameter specifications. Defaults to None.
- `list_params`: The list of strs of parameter names.
- `model_name`: The name of the model.
- `model_config`: A dictionary representing the model configuration.
- `model_distribution`: The likelihood function of the model in the form of a pm.Distribution subclass.
- `family`: A Bambi family object.
- `priors`: A dictionary containing the prior distribution of parameters.
- `formula`: A string representing the model formula.
- `link`: A string or a dictionary representing the link functions for all parameters.
- `params`: A list of Param objects representing model parameters.

### Methods

- `__init__`: Constructor method that initializes the HSSM class.
- `sample`: A method to sample posterior distributions.
- `set_alias`: Sets the alias for a parameter.
- `graph`: Plot the model with PyMC's built-in graph function.

### Method: `__init__`

#### Description

Constructor method that initializes the HSSM class.

#### Parameters

- `data`: A pandas DataFrame.
- `model`: SupportedModels, defaults to "ddm".
- `include`: A list of dictionaries or None, defaults to None.
- `model_config`: Config or None, defaults to None.
- `loglik`: LogLikeFunc or pytensor.graph.Op or None, defaults to None.
- `**kwargs`: Other parameters.

### Method: `_transform_params`

Transforms a list of dictionaries containing parameter information into a list of Param objects. It also generates a formula, priors, and a link for the Bambi package based on these parameters.

#### Parameters

- **include**: A list of dictionaries with details about the parameters.
- **model**: A string indicating the type of the model.
- **model_config**: A dict providing configuration details for the model.

#### Returns

A tuple of four items:

1. A list of Param objects.
2. A Bambi formula object.
3. An optional dict containing prior information for Bambi.
4. An optional dict of link functions for Bambi.

### Method: `Sample`

Performs sampling using the `fit` method via the Bambi Model.

#### Parameters

- **sampler**: The sampler to use. Options include "mcmc" (default), "nuts_numpyro", "nuts_blackjax", "laplace", or "vi".
- **kwargs**: Other arguments passed to Bambi Model's `fit` method.

#### Returns

An ArviZ `InferenceData` instance if `inference_method` is "mcmc", "nuts_numpyro", "nuts_blackjax" or "laplace". An `Approximation` object if "vi".

### pymc_model Property

Returns the PyMC model built by Bambi.

#### Returns

The PyMC model built by Bambi.

### Method: `set_alias`

Sets the aliases according to the dictionary passed



::: hssm.hssm